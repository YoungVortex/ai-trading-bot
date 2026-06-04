# backtester.py
import numpy as np
import pandas as pd
import torch
from config import *
from smc_engine import smc_analysis, smc_feature_vector

class Portfolio:
    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = 0
        self.equity_curve = []
        self.trades = []

    def update(self, fill_price, direction, quantity, timestamp):
        if direction == 'LONG':
            cost = fill_price * quantity
            if cost <= self.cash:
                self.positions += quantity
                self.cash -= cost
                self.trades.append({
                    'timestamp': str(timestamp), 'type': 'BUY',
                    'price': fill_price, 'quantity': quantity, 'pnl': 0
                })
        elif direction == 'SHORT':
            if self.positions >= quantity:
                proceeds = fill_price * quantity
                avg_entry = (
                    sum(t['price']*t['quantity'] for t in self.trades if t['type']=='BUY') / self.positions
                    if self.positions > 0 else 0
                )
                pnl = (fill_price - avg_entry) * quantity
                self.cash += proceeds
                self.positions -= quantity
                self.trades.append({
                    'timestamp': str(timestamp), 'type': 'SELL',
                    'price': fill_price, 'quantity': quantity, 'pnl': pnl
                })
        mark_price = fill_price
        equity = self.cash + self.positions * mark_price
        self.equity_curve.append(equity)

    def get_metrics(self):
        eq = pd.Series(self.equity_curve)
        if len(eq) < 2:
            return {'win_rate':0, 'profit_factor':0, 'sharpe_ratio':0,
                    'max_drawdown':0, 'final_equity': self.initial_capital}
        returns = eq.pct_change().dropna()
        closed_trades = [t for t in self.trades if t['type']=='SELL']
        wins = [t['pnl'] for t in closed_trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in closed_trades if t['pnl'] <= 0]
        total_trades = len(closed_trades)
        win_rate = len(wins)/total_trades if total_trades else 0
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-8
        profit_factor = gross_profit/gross_loss if gross_loss else float('inf')
        sharpe = np.sqrt(252) * returns.mean()/returns.std() if returns.std() else 0
        max_dd = (eq.cummax() - eq).max() / eq.cummax().max()
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_equity': eq.iloc[-1]
        }

class Backtester:
    def __init__(self, data, model, initial_capital=INITIAL_CAPITAL,
                 risk_per_trade=RISK_PER_TRADE, use_smc=True, device=DEVICE):
        self.data = data
        self.model = model
        self.model.eval()
        self.device = device
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.risk_per_trade = risk_per_trade
        self.use_smc = use_smc

    def _prepare_input(self, window_df):
        ohlcv = window_df[['open','high','low','close']].values.astype(np.float32)
        ohlcv = ohlcv / ohlcv[0] - 1.0
        x = torch.tensor(ohlcv).unsqueeze(0).to(self.device)
        ohlcv_dict = {tf: x for tf in TIMEFRAMES}
        smc_feat = None
        if self.use_smc:
            smc_res = smc_analysis(window_df)
            smc_feat = torch.tensor(smc_feature_vector(smc_res, window_size=len(window_df)),
                                    dtype=torch.float32).unsqueeze(0).to(self.device)
        return ohlcv_dict, smc_feat

    def run(self, start_idx=None, end_idx=None):
        bars = self.data
        start_idx = start_idx or SEQ_LEN
        end_idx = end_idx or len(bars)-1

        for i in range(start_idx, end_idx):
            window_df = bars.iloc[i-SEQ_LEN:i].copy()
            ohlcv_dict, smc_feat = self._prepare_input(window_df)
            with torch.no_grad():
                logits, confidence = self.model(ohlcv_dict=ohlcv_dict, smc_features=smc_feat)
                probs = torch.softmax(logits, dim=-1)
                pred_class = torch.argmax(probs, dim=1).item()
                conf = confidence.item()

            direction = {0:'LONG', 1:'SHORT', 2:'NONE'}[pred_class]
            price = bars.iloc[i]['close']
            atr = self._atr(bars, i)
            if atr == 0:
                atr = price * 0.01
            risk_amount = self.portfolio.cash * self.risk_per_trade
            stop_loss_dist = atr * SL_ATR_MULT
            qty = int(risk_amount / stop_loss_dist) if stop_loss_dist > 0 else 0

            if direction == 'LONG' and conf > 60 and qty > 0:
                self.portfolio.update(price, 'LONG', qty, bars.index[i])
            elif direction == 'SHORT' and conf > 60 and self.portfolio.positions > 0:
                self.portfolio.update(price, 'SHORT', self.portfolio.positions, bars.index[i])

        return self.portfolio.get_metrics()

    def _atr(self, bars, idx, period=14):
        if idx < period:
            period = idx
        high = bars['high'].iloc[idx-period:idx]
        low = bars['low'].iloc[idx-period:idx]
        close = bars['close'].iloc[idx-period:idx]
        if len(high) < 2:
            return 0
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.abs(high - prev_close), np.abs(low - prev_close))
        return tr.mean()
