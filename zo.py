# zo.py
import sys
import json
import numpy as np
import pandas as pd
import torch
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from config import *
from utils import generate_persian_reasoning
from smc_engine import smc_analysis, smc_feature_vector
from model import MarketPredictor
from backtester import Backtester

class TradingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Trading Terminal – Spot $1000")
        self.setGeometry(100, 100, 1500, 900)
        self.model = None
        self.data = None
        self._init_ui()
        self._apply_stylesheet()

    def _init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Left panel
        left_panel = QtWidgets.QVBoxLayout()
        self.load_btn = QtWidgets.QPushButton("بارگذاری CSV")
        self.load_btn.clicked.connect(self.load_data)
        left_panel.addWidget(self.load_btn)

        self.model_path_edit = QtWidgets.QLineEdit("market_model.pt")
        left_panel.addWidget(QtWidgets.QLabel("مسیر مدل:"))
        left_panel.addWidget(self.model_path_edit)

        self.load_model_btn = QtWidgets.QPushButton("بارگذاری مدل")
        self.load_model_btn.clicked.connect(self.load_model)
        left_panel.addWidget(self.load_model_btn)

        self.predict_btn = QtWidgets.QPushButton("دریافت سیگنال")
        self.predict_btn.clicked.connect(self.predict)
        left_panel.addWidget(self.predict_btn)

        self.signal_text = QtWidgets.QTextEdit()
        self.signal_text.setReadOnly(True)
        left_panel.addWidget(QtWidgets.QLabel("سیگنال و تحلیل:"))
        left_panel.addWidget(self.signal_text)

        self.backtest_btn = QtWidgets.QPushButton("اجرای بک‌تست (Spot $1000)")
        self.backtest_btn.clicked.connect(self.run_backtest)
        left_panel.addWidget(self.backtest_btn)

        self.metrics_text = QtWidgets.QTextEdit()
        self.metrics_text.setReadOnly(True)
        left_panel.addWidget(QtWidgets.QLabel("نتایج بک‌تست:"))
        left_panel.addWidget(self.metrics_text)

        left_panel.addStretch()
        main_layout.addLayout(left_panel, 1)

        # Right panel – chart
        right_panel = QtWidgets.QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        right_panel.addWidget(self.canvas)
        main_layout.addLayout(right_panel, 3)

    def _apply_stylesheet(self):
        dark_style = """
        QMainWindow { background-color: #1e1e2e; }
        QPushButton {
            background-color: #2d2d44; color: #cdd6f4; border: 1px solid #45475a;
            padding: 8px; border-radius: 4px; font-weight: bold;
        }
        QPushButton:hover { background-color: #45475a; }
        QTextEdit, QLineEdit { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; }
        QLabel { color: #cdd6f4; }
        """
        self.setStyleSheet(dark_style)

    def load_data(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        self.data = pd.read_csv(path)
        QtWidgets.QMessageBox.information(self, "موفق", f"{len(self.data)} کندل بارگذاری شد")
        self.plot_chart()

    def load_model(self):
        path = self.model_path_edit.text()
        try:
            self.model = MarketPredictor(use_image=False).to(DEVICE)
            self.model.load_state_dict(torch.load(path, map_location=DEVICE))
            self.model.eval()
            QtWidgets.QMessageBox.information(self, "موفق", "مدل بارگذاری شد")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "خطا", str(e))

    def predict(self):
        if self.data is None or self.model is None:
            QtWidgets.QMessageBox.warning(self, "اخطار", "ابتدا داده و مدل را بارگذاری کنید")
            return
        if len(self.data) < SEQ_LEN:
            QtWidgets.QMessageBox.warning(self, "اخطار", f"حداقل {SEQ_LEN} کندل نیاز است")
            return
        window_df = self.data.iloc[-SEQ_LEN:].copy()
        ohlcv = window_df[['open','high','low','close']].values.astype(np.float32)
        ohlcv = ohlcv / ohlcv[0] - 1.0
        x = torch.tensor(ohlcv).unsqueeze(0).to(DEVICE)
        ohlcv_dict = {tf: x for tf in TIMEFRAMES}
        smc_res = smc_analysis(window_df)
        smc_feat = torch.tensor(smc_feature_vector(smc_res, len(window_df)),
                                dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, confidence = self.model(ohlcv_dict=ohlcv_dict, smc_features=smc_feat)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=1).item()
            conf = confidence.item()
        direction = {0:'LONG', 1:'SHORT', 2:'NO_TRADE'}[pred]
        trend = "bullish" if pred == 0 else "bearish" if pred == 1 else "neutral"
        reason = generate_persian_reasoning(direction, conf, smc_res, trend)
        self.signal_text.setText(reason)

    def run_backtest(self):
        if self.data is None or self.model is None:
            QtWidgets.QMessageBox.warning(self, "اخطار", "ابتدا داده و مدل را بارگذاری کنید")
            return
        bt = Backtester(self.data, self.model, initial_capital=1000.0)
        metrics = bt.run()
        self.metrics_text.setText(json.dumps(metrics, indent=2, ensure_ascii=False))
        # Plot equity curve
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(bt.portfolio.equity_curve)
        ax.set_title("Equity Curve")
        self.canvas.draw()

    def plot_chart(self):
        if self.data is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.data['close'])
        ax.set_title("Price Chart")
        self.canvas.draw()