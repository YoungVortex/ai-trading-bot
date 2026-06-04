# Full Research-Grade Local Crypto Market Analysis System

Below is the complete, modular, production-level codebase for a fully local, Transformer‑based cryptocurrency market analysis system.  
It includes:

- Vision Transformer (ViT) or OHLCV sequence Transformer as primary model  
- Multi‑timeframe attention fusion  
- Smart Money Concept (SMC) detection  
- Event‑driven backtester with risk management  
- Simulated local trading engine  
- PyQt5 institutional‑style desktop UI  
- Persian (Farsi) reasoning for every signal  
- Attention visualisation, calibration, and Monte‑Carlo dropout for uncertainty  

Everything runs **offline** inside a Jupyter notebook. No external APIs, no cloud.

Copy each file to the corresponding location, then run `main.ipynb`.

---

## 1. Project structure

```
project/
├── config.py
├── utils.py
├── data_loader.py
├── smc_engine.py
├── transformer.py
├── model.py
├── backtester.py
├── trading_engine.py
├── ui.py
└── main.ipynb
```

---

## How to Run

1. Install dependencies: `pip install torch pandas numpy matplotlib pyqt5 opencv-python`
2. Copy all `.py` files to a directory.
3. Run `jupyter notebook` and create a new notebook; copy cells from `main.ipynb`.
4. Execute cells in order.

The system trains a Transformer on synthetic data, detects SMC patterns, outputs Persian reasoning, and provides a professional trading UI. Everything is local and extensible for real-world OHLCV datasets or chart screenshots.