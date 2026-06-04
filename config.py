# config.py
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
SEQ_LEN = 64          # OHLCV lookback window
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
PATCH_SIZE = 16       # for ViT
IMAGE_SIZE = 224
NUM_CLASSES = 3       # LONG, SHORT, NO TRADE

# Training
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20
MC_DROPOUT_SAMPLES = 30  # for uncertainty

# SMC
SWING_STRENGTH = 5       # candles to define swing points
ORDER_BLOCK_LOOKBACK = 3
LIQUIDITY_SWEEP_THRESH = 0.005  # 0.5%

# Trading / Backtest
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE = 0.02     # 2%
MAX_DRAWDOWN = 0.25       # 25% stop
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.0

# Multi-timeframe
TIMEFRAMES = ["1m", "5m", "15m", "1h"]