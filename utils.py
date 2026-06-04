# utils.py
import numpy as np
import random

def set_seed(seed=42):
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_persian_reasoning(prediction, confidence, smc_signals, trend):
    decision_fa = {"LONG": "خرید (LONG)", "SHORT": "فروش (SHORT)", "NO_TRADE": "عدم معامله"}
    reason = f"تصمیم: {decision_fa.get(prediction, prediction)}\n"
    reason += f"اعتماد: {confidence:.0f}%\n"
    reason += "دلیل:\n"
    if trend == "bullish":
        reason += "- روند کلی بازار صعودی است.\n"
    elif trend == "bearish":
        reason += "- روند کلی بازار نزولی است.\n"
    else:
        reason += "- بازار در حالت رنج و بدون روند مشخص است.\n"
    if smc_signals.get("bos"):
        reason += "- شکست ساختار (BOS) تشخیص داده شد.\n"
    if smc_signals.get("choch"):
        reason += "- تغییر کاراکتر (CHoCH) رخ داده است.\n"
    if smc_signals.get("order_blocks"):
        ob = smc_signals["order_blocks"][-1]
        reason += f"- اردر بلاک معتبر در {ob['low']:.2f} تا {ob['high']:.2f}.\n"
    if smc_signals.get("liquidity_sweep"):
        reason += "- جاروکردن نقدینگی (Liquidity Sweep) شناسایی شد.\n"
    if confidence > 80:
        reason += "- مدل با اطمینان بالا این سیگنال را تأیید می‌کند.\n"
    else:
        reason += "- مدیریت ریسک را فراموش نکنید.\n"
    return reason