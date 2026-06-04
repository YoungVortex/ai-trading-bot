# trading_engine.py
import json
import time
from datetime import datetime
from config import *

class LocalTradingEngine:
    def __init__(self, model, portfolio=None):
        self.model = model
        self.portfolio = portfolio if portfolio else {'cash': INITIAL_CAPITAL, 'positions': {}}
        self.trade_journal = []
        self.is_running = False

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False

    def on_new_bar(self, ohlcv_dict, smc_features):
        if not self.is_running:
            return
        # Get prediction
        with torch.no_grad():
            logits, conf = self.model(ohlcv_dict=ohlcv_dict, smc_features=smc_features)
            pred = torch.argmax(logits, dim=1).item()
            direction = {0:'LONG', 1:'SHORT', 2:'NONE'}[pred]
        
        if direction != 'NONE' and conf > 60:
            # Simulate order
            trade = {
                'timestamp': str(datetime.now()),
                'direction': direction,
                'confidence': conf.item(),
                'status': 'executed',
                'price': ohlcv_dict['1m'][:,-1,3].item(),   # last close
                'quantity': 1  # simplified
            }
            self.trade_journal.append(trade)
            # Log to file
            with open('trade_journal.json', 'a') as f:
                f.write(json.dumps(trade) + '\n')