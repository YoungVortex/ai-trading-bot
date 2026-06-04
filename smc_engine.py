# smc_engine.py
import numpy as np
import pandas as pd
from config import SWING_STRENGTH, ORDER_BLOCK_LOOKBACK, LIQUIDITY_SWEEP_THRESH

def detect_swing_points(high, low, strength=SWING_STRENGTH):
    """Find swing highs and lows."""
    swing_highs = []
    swing_lows = []
    for i in range(strength, len(high)-strength):
        if all(high[i] >= high[i-strength:i]) and all(high[i] >= high[i+1:i+strength+1]):
            swing_highs.append(i)
        if all(low[i] <= low[i-strength:i]) and all(low[i] <= low[i+1:i+strength+1]):
            swing_lows.append(i)
    return swing_highs, swing_lows

def detect_bos_choch(high, low, swing_highs, swing_lows):
    """Break of Structure and Change of Character."""
    bos = []
    choch = []
    # BOS: price breaks previous swing high in uptrend or low in downtrend
    for i in range(len(swing_highs)-1):
        if high[swing_highs[i+1]] > high[swing_highs[i]]:
            bos.append(swing_highs[i+1])
    for i in range(len(swing_lows)-1):
        if low[swing_lows[i+1]] < low[swing_lows[i]]:
            bos.append(swing_lows[i+1])
    # CHoCH: opposite break
    for idx in swing_highs:
        if idx > 0 and low[idx] < low[idx-1]:   # simplified
            choch.append(idx)
    for idx in swing_lows:
        if idx > 0 and high[idx] > high[idx-1]:
            choch.append(idx)
    return bos, choch

def detect_order_blocks(open, high, low, close):
    """Order blocks: last opposite candle before a strong move."""
    obs = []
    for i in range(ORDER_BLOCK_LOOKBACK, len(close)-1):
        move = close[i+1] - close[i]
        if abs(move) > 0.01 * close[i]:   # 1% move
            # bullish OB
            if move > 0 and close[i] < open[i]:
                obs.append({'idx': i, 'type': 'bullish', 'high': high[i], 'low': low[i]})
            # bearish OB
            elif move < 0 and close[i] > open[i]:
                obs.append({'idx': i, 'type': 'bearish', 'high': high[i], 'low': low[i]})
    return obs

def detect_liquidity_sweeps(high, low):
    """Liquidity sweeps: equal highs/lows get broken slightly."""
    sweeps = []
    for i in range(1, len(high)):
        if high[i] > high[i-1] and abs(high[i]-high[i-1]) < LIQUIDITY_SWEEP_THRESH * high[i]:
            sweeps.append({'idx': i, 'type': 'high_sweep'})
        if low[i] < low[i-1] and abs(low[i]-low[i-1]) < LIQUIDITY_SWEEP_THRESH * low[i]:
            sweeps.append({'idx': i, 'type': 'low_sweep'})
    return sweeps

def smc_analysis(df):
    """Run full SMC analysis on OHLCV DataFrame."""
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    
    swing_highs, swing_lows = detect_swing_points(h, l)
    bos, choch = detect_bos_choch(h, l, swing_highs, swing_lows)
    order_blocks = detect_order_blocks(o, h, l, c)
    sweeps = detect_liquidity_sweeps(h, l)
    
    return {
        'swing_highs': swing_highs,
        'swing_lows': swing_lows,
        'bos': bos,
        'choch': choch,
        'order_blocks': order_blocks,
        'liquidity_sweeps': sweeps,
    }

def smc_feature_vector(smc_result, window_size=64):
    """Encode SMC events as a fixed-size vector for model input."""
    vec = []
    # Counts normalised
    vec.append(len(smc_result['bos']) / window_size)
    vec.append(len(smc_result['choch']) / window_size)
    vec.append(len(smc_result['order_blocks']) / window_size)
    vec.append(len(smc_result['liquidity_sweeps']) / window_size)
    # Last order block zone (if any)
    obs = smc_result['order_blocks']
    if obs:
        ob = obs[-1]
        vec.append((ob['high'] + ob['low'])/2)   # mean price of OB
    else:
        vec.append(0.0)
    # Trend: difference between last swing highs/lows
    sh = smc_result['swing_highs']
    sl = smc_result['swing_lows']
    if len(sh) >= 2 and len(sl) >= 2:
        trend_val = (sh[-1] - sh[-2]) + (sl[-1] - sl[-2])
        vec.append(trend_val / window_size)
    else:
        vec.append(0.0)
    return np.array(vec, dtype=np.float32)