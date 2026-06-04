# backtester.py
from collections import deque
import numpy as np
import pandas as pd
from datetime import datetime
from config import *
import json

class Event:
    pass

class MarketEvent(Event):
    def __init__(self, bar):
        self.bar = bar

class SignalEvent(Event):
    def __init__(self, direction, confidence):
        self.direction = direction   # 'LONG', 'SHORT', 'NONE'
        self.confidence = confidence

class OrderEvent(Event):
    def __init__(self, symbol, order_type, quantity, direction):
        self.symbol = symbol
        self.type = order_type
        self.quantity = quantity
        self.direction = direction

class FillEvent(Event):
    def __init__(self, timestamp, symbol, direction, quantity, price, commission=0.0):
        self.timestamp = timestamp
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.price = price
        self.commission = commission

class Portfolio:
    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}   # symbol -> quantity
        self.equity_curve = []
        self.trades = []

    def update(self, fill):
        if fill.direction == 'LONG':
            self.positions[fill.symbol] = self.positions.get(fill.symbol, 0) + fill.quantity
            self.cash -= fill.price * fill.quantity + fill.commission
        elif fill.direction == 'SHORT':
            self.positions[fill.symbol] = self.positions.get(fill.symbol, 0) - fill.quantity
            self.cash += fill.price * fill.quantity - fill.commission

        self.trades.append(fill.__dict__)
        current_equity = self.cash + sum(pos * fill.price for sym, pos in self.positions.items())
        self.equity_curve.append(current_equity)

    def get_metrics(self):
        eq = pd.Series(self.equity_curve)
        returns = eq.pct_change().dropna()
        win_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        total_trades = len(self.trades)
        win_rate = len(win_trades)/total_trades if total_trades>0 else 0
        gross_profit = sum(t.get('pnl',0) for t in self.trades if t.get('pnl',0)>0)
        gross_loss = abs(sum(t.get('pnl',0) for t in self.trades if t.get('pnl',0)<0))
        profit_factor = gross_profit/gross_loss if gross_loss>0 else float('inf')
        sharpe = np.sqrt(252) * returns.mean()/returns.std() if returns.std()>0 else 0
        max_dd = (eq.cummax() - eq).max() / eq.cummax().max()
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_equity': self.equity_curve[-1] if self.equity_curve else self.initial_capital
        }

class RiskManager:
    def __init__(self, portfolio, max_drawdown=MAX_DRAWDOWN, risk_per_trade=RISK_PER_TRADE):
        self.portfolio = portfolio
        self.max_dd = max_drawdown
        self.risk_per_trade = risk_per_trade

    def size_position(self, price, atr):
        risk_amount = self.portfolio.cash * self.risk_per_trade
        quantity = risk_amount / (atr * SL_ATR_MULT) if atr>0 else 0
        return int(quantity)

class Backtester:
    def __init__(self, data, model, smc_engine_func=None):
        self.data = data
        self.model = model
        self.smc_func = smc_engine_func
        self.events = deque()
        self.portfolio = Portfolio()
        self.risk_mgr = RiskManager(self.portfolio)
        self.signal_generated = False

    def run(self):
        bars = self.data
        for i in range(SEQ_LEN, len(bars)-1):
            # Get OHLCV window
            window = bars.iloc[i-SEQ_LEN:i]
            # Generate signal (simulate model inference)
            # In practice, call model.predict() - here we use dummy
            signal = self._generate_signal(window)
            if signal:
                # Execute
                price = bars.iloc[i]['close']
                atr = self._atr(bars, i)
                qty = self.risk_mgr.size_position(price, atr)
                if qty > 0:
                    fill = FillEvent(bars.index[i], 'BTCUSDT', signal, qty, price)
                    self.portfolio.update(fill)
        return self.portfolio.get_metrics()

    def _generate_signal(self, window):
        # Placeholder: random signal for demonstration
        # In real system, use model.predict
        return np.random.choice(['LONG','SHORT','NONE'], p=[0.33,0.33,0.34])

    def _atr(self, bars, idx, period=14):
        high = bars['high'].iloc[idx-period:idx]
        low = bars['low'].iloc[idx-period:idx]
        close = bars['close'].iloc[idx-period:idx]
        tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
        return tr.mean()