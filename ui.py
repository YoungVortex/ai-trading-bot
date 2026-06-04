# ui.py
import sys
import json
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class TradingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Trading Terminal – SMC & Transformer")
        self.setGeometry(100, 100, 1400, 900)
        self._init_ui()
        self._apply_stylesheet()

    def _init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Left panel - controls
        left_panel = QtWidgets.QVBoxLayout()
        self.upload_btn = QtWidgets.QPushButton("بارگذاری چارت یا CSV")
        self.upload_btn.clicked.connect(self.load_data)
        left_panel.addWidget(self.upload_btn)

        self.timeframe_combo = QtWidgets.QComboBox()
        self.timeframe_combo.addItems(["1m", "5m", "15m", "1h"])
        left_panel.addWidget(QtWidgets.QLabel("تایم‌فریم:"))
        left_panel.addWidget(self.timeframe_combo)

        self.predict_btn = QtWidgets.QPushButton("دریافت سیگنال")
        self.predict_btn.clicked.connect(self.predict)
        left_panel.addWidget(self.predict_btn)

        self.signal_text = QtWidgets.QTextEdit()
        self.signal_text.setReadOnly(True)
        left_panel.addWidget(QtWidgets.QLabel("سیگنال و تحلیل:"))
        left_panel.addWidget(self.signal_text)

        self.backtest_btn = QtWidgets.QPushButton("اجرای بک‌تست")
        left_panel.addWidget(self.backtest_btn)

        left_panel.addStretch()
        main_layout.addLayout(left_panel, 1)

        # Right panel - chart + metrics
        right_panel = QtWidgets.QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        right_panel.addWidget(self.canvas)

        self.metrics_table = QtWidgets.QTableWidget()
        right_panel.addWidget(self.metrics_table)

        main_layout.addLayout(right_panel, 3)

    def _apply_stylesheet(self):
        dark_style = """
        QMainWindow { background-color: #1e1e2e; }
        QPushButton {
            background-color: #2d2d44; color: #cdd6f4; border: 1px solid #45475a;
            padding: 8px; border-radius: 4px; font-weight: bold;
        }
        QPushButton:hover { background-color: #45475a; }
        QTextEdit, QTableWidget { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; }
        QLabel { color: #cdd6f4; }
        QComboBox { background-color: #313244; color: #cdd6f4; }
        """
        self.setStyleSheet(dark_style)

    def load_data(self):
        # Mock: just show a message
        QtWidgets.QMessageBox.information(self, "Info", "داده‌ها از فایل CSV بارگذاری شد (شبیه‌سازی)")

    def predict(self):
        # Mock prediction with Persian reasoning
        decision = "LONG"
        confidence = 84
        reason = ("دلیل:\n"
                  "- روند صعودی قوی با BOS تأیید شده\n"
                  "- یک اردر بلاک صعودی در ۴۵۲۰۰ دلار\n"
                  "- جاروب نقدینگی بالای مقاومت قبلی")
        full = f"تصمیم: {decision}\nاعتماد: {confidence}%\n{reason}"
        self.signal_text.setText(full)

    def show_chart(self, data):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(data)
        self.canvas.draw()