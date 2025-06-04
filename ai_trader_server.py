import sys
import os

# Thêm thư mục gốc vào sys.path để đảm bảo Python tìm thấy module ai_trader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import zmq
import json
import pandas as pd
import numpy as np
import backtrader as bt
import logging
from datetime import datetime
from ai_trader.trader import AITrader
from ai_trader.strategy.base import BaseStrategy
from ai_trader.strategy.sma import NaiveSMAStrategy, CrossSMAStrategy
from ai_trader.strategy.bbands import BBandsStrategy
from ai_trader.strategy.momentum import MomentumStrategy
from ai_trader.strategy.rsi import RsiBollingerBandsStrategy, TripleRsiStrategy
from ai_trader.strategy.rsrs import RSRSStrategy
from ai_trader.strategy.roc import ROCStochStrategy, ROCMAStrategy, NaiveROCStrategy
from ai_trader.strategy.double_top import DoubleTopStrategy
from ai_trader.strategy.turtle import TurtleTradingStrategy
from ai_trader.strategy.vcp import VCPStrategy
from ai_trader.strategy.roc_rotation import ROCRotationStrategy
from ai_trader.strategy.rsrs_rotation import RSRSRotationStrategy
from ai_trader.strategy.triple_rsi import TripleRSIRotationStrategy
from ai_trader.strategy.multi_bbands import MultiBBandsRotationStrategy
from ai_trader.ensemble_model import EnsembleModel
from sklearn.preprocessing import MinMaxScaler
import cv2
import base64
from io import BytesIO
from PIL import Image

# Thiết lập logging để hiển thị trên cả console và file
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Formatter cho log
formatter = logging.Formatter("[%(asctime)s] [AI] %(message)s")

# Handler cho console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Handler cho file
file_handler = logging.FileHandler(
    f"C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\data\\ai_server_{datetime.now().strftime('%Y%m%d')}.log"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Danh sách chiến lược
STRATEGIES = [
    ("NaiveSMAStrategy", NaiveSMAStrategy),
    ("CrossSMAStrategy", CrossSMAStrategy),
    ("BBandsStrategy", BBandsStrategy),
    ("MomentumStrategy", MomentumStrategy),
    ("RsiBollingerBandsStrategy", RsiBollingerBandsStrategy),
    ("TripleRsiStrategy", TripleRsiStrategy),
    ("RSRSStrategy", RSRSStrategy),
    ("ROCStochStrategy", ROCStochStrategy),
    ("ROCMAStrategy", ROCMAStrategy),
    ("NaiveROCStrategy", NaiveROCStrategy),
    ("DoubleTopStrategy", DoubleTopStrategy),
    ("TurtleTradingStrategy", TurtleTradingStrategy),
    ("VCPStrategy", VCPStrategy),
    ("ROCRotationStrategy", ROCRotationStrategy),
    ("RSRSRotationStrategy", RSRSRotationStrategy),
    ("TripleRSIRotationStrategy", TripleRSIRotationStrategy),
    ("MultiBBandsRotationStrategy", MultiBBandsRotationStrategy)
]

class CustomDataFeed(bt.feeds.PandasData):
    """
    Định nghĩa lớp dữ liệu tùy chỉnh cho Backtrader.
    """
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),
    )

# Kiểm tra sự tồn tại của các file mô hình và scaler
required_files = [
    "C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\lstm_model.keras",
    "C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\lstm_scaler.npy",
    "C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\cnn_model.keras",
    "C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\cnn_scaler.npy",
    "C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\tcn_model.keras",
    "C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\tcn_scaler.npy",
    "C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\xgboost_model.json",
    "C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\xgboost_scaler.npy"
]

for file_path in required_files:
    if not os.path.exists(file_path):
        logging.error(f"Thiếu file: {file_path}. Vui lòng chạy các file huấn luyện để tạo file mô hình và scaler.")
        raise FileNotFoundError(f"Thiếu file: {file_path}. Vui lòng chạy các file huấn luyện để tạo file mô hình và scaler.")

# Load scaler (load toàn bộ đối tượng scaler)
try:
    logging.info("Bắt đầu load scaler")
    lstm_scaler = np.load("C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\lstm_scaler.npy", allow_pickle=True).item()
    cnn_scaler = np.load("C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\cnn_scaler.npy", allow_pickle=True).item()
    tcn_scaler = np.load("C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\tcn_scaler.npy", allow_pickle=True).item()
    xgb_scaler = np.load("C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\xgboost_scaler.npy", allow_pickle=True).item()
    logging.info("Load scaler thành công")
except Exception as e:
    logging.error(f"Lỗi load scaler: {e}")
    raise Exception(f"Lỗi load scaler: {e}")

# Load ensemble model
try:
    logging.info("Bắt đầu load ensemble model")
    ensemble_model = EnsembleModel()
    logging.info("Load ensemble model thành công")
except Exception as e:
    logging.error(f"Lỗi load ensemble model: {e}")
    raise Exception(f"Lỗi load ensemble model: {e}")

def calculate_technical_indicators(data, timesteps=28):
    """
    Tính 28 chỉ báo kỹ thuật dựa trên dữ liệu OHLC và Volume.
    
    Parameters:
    - data: Dữ liệu OHLC và Volume (numpy array) với shape (samples, 5).
    - timesteps: Số timesteps để tính các chỉ báo.
    
    Returns:
    - Dữ liệu mở rộng với 28 đặc trưng (numpy array) với shape (samples, 28).
    """
    try:
        logging.info("Bắt đầu tính các chỉ báo kỹ thuật")
        df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # 1. Yield (Tỷ suất thay đổi giá)
        df['Yield'] = (df['Close'] - df['Open']) / df['Open']
        
        # 2. Percentage Volume
        df['PercVol'] = df['Volume'].pct_change().fillna(0)
        
        # 3-6. Simple Moving Averages (SMA) với các kỳ khác nhau
        df['SMA5'] = df['Close'].rolling(window=5).mean()
        df['SMA10'] = df['Close'].rolling(window=10).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        # 7-10. Exponential Moving Averages (EMA)
        df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # 11-14. Weighted Moving Averages (WMA)
        def wma(series, window):
            weights = np.arange(1, window + 1)
            return series.rolling(window).apply(lambda x: np.sum(x * weights) / weights.sum(), raw=True)
        
        df['WMA5'] = wma(df['Close'], 5)
        df['WMA10'] = wma(df['Close'], 10)
        df['WMA20'] = wma(df['Close'], 20)
        df['WMA50'] = wma(df['Close'], 50)
        
        # 15-18. Hull Moving Averages (HMA)
        def hma(series, period):
            wma_half = wma(series, period // 2)
            wma_full = wma(series, period)
            raw_hma = 2 * wma_half - wma_full
            return wma(raw_hma, int(np.sqrt(period)))
        
        df['HMA5'] = hma(df['Close'], 5)
        df['HMA10'] = hma(df['Close'], 10)
        df['HMA20'] = hma(df['Close'], 20)
        df['HMA50'] = hma(df['Close'], 50)
        
        # 19. MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        
        # 20. Commodity Channel Index (CCI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mean_dev = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df['CCI'] = (typical_price - sma_tp) / (0.015 * mean_dev)
        
        # 21. Stochastic Oscillator (%K)
        lowest_low = df['Low'].rolling(window=14).min()
        highest_high = df['High'].rolling(window=14).max()
        df['StochOsc'] = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
        
        # 22. Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 23. Rate of Change (ROC)
        df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
        
        # 24. Percentage Price Oscillator (PPO)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['PPO'] = ((ema12 - ema26) / ema26) * 100
        
        # 25. Know Sure Thing (KST)
        roc1 = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        roc2 = ((df['Close'] - df['Close'].shift(15)) / df['Close'].shift(15)) * 100
        roc3 = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)) * 100
        roc4 = ((df['Close'] - df['Close'].shift(30)) / df['Close'].shift(30)) * 100
        df['KST'] = (roc1 * 1) + (roc2 * 2) + (roc3 * 3) + (roc4 * 4)
        
        # 26-28. Bollinger Bands
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['BOLM'] = sma20
        df['BOLU'] = sma20 + (2 * std20)
        df['BOLD'] = sma20 - (2 * std20)
        
        # Điền giá trị NaN
        df = df.fillna(0)
        
        # Chuyển thành numpy array với 28 đặc trưng
        return df[['Yield', 'PercVol', 'SMA5', 'SMA10', 'SMA20', 'SMA50', 'EMA5', 'EMA10', 'EMA20', 'EMA50',
                   'WMA5', 'WMA10', 'WMA20', 'WMA50', 'HMA5', 'HMA10', 'HMA20', 'HMA50', 'MACD', 'CCI',
                   'StochOsc', 'RSI', 'ROC', 'PPO', 'KST', 'BOLM', 'BOLU', 'BOLD']].values
    except Exception as e:
        logging.error(f"Lỗi khi tính chỉ báo kỹ thuật: {e}")
        raise

def preprocess_data(data_json, lookback=28):
    """
    Tiền xử lý dữ liệu từ EA để sử dụng cho Backtrader và các mô hình máy học.
    
    Parameters:
    - data_json: Dữ liệu JSON từ EA.
    - lookback: Số timesteps để tạo sequence.
    
    Returns:
    - df: DataFrame cho Backtrader.
    - X_lstm: Dữ liệu đầu vào cho LSTM.
    - X_cnn: Dữ liệu OHLC cho CNN.
    - X_image: Ảnh chart cho CNN.
    - X_tcn: Dữ liệu đầu vào cho TCN.
    - X_xgb: Dữ liệu đầu vào cho XGBoost.
    """
    try:
        logging.info("Bắt đầu tiền xử lý dữ liệu")
        
        # Xử lý dữ liệu OHLC
        ohlc = np.array(data_json["ohlc"])
        rsi = np.array(data_json["rsi"])
        ma20 = np.array(data_json["ma20"])
        ma50 = np.array(data_json["ma50"])
        bb_upper = np.array(data_json["bollinger"]["upper"])
        bb_lower = np.array(data_json["bollinger"]["lower"])
        bb_middle = np.array(data_json["bollinger"]["middle"])
        macd = np.array(data_json["macd"]["macd"])
        macd_signal = np.array(data_json["macd"]["signal"])
        macd_hist = np.array(data_json["macd"]["hist"])
        
        # Tạo DataFrame cho Backtrader
        df = pd.DataFrame(ohlc, columns=["Open", "High", "Low", "Close"])
        df["Volume"] = 1000  # Giả lập volume
        df["Date"] = pd.date_range(start="2025-05-31 09:00:00", periods=len(ohlc), freq="5min")
        df.set_index("Date", inplace=True)
        logging.info("Tạo DataFrame cho Backtrader thành công")

        # Tạo dữ liệu ban đầu cho các chỉ báo kỹ thuật
        data = np.column_stack([ohlc, np.ones((len(ohlc), 1)) * 1000])
        
        # Tính các chỉ báo kỹ thuật
        indicators = calculate_technical_indicators(data, lookback)
        
        # Tạo dữ liệu cho các mô hình máy học
        data_ml = np.column_stack([
            indicators  # 28 đặc trưng từ chỉ báo kỹ thuật
        ])
        
        # Tạo dữ liệu mở rộng cho XGBoost
        data_xgb = np.column_stack([
            indicators,
            (data[:, 3] - data[:, 0]).reshape(-1, 1),  # PriceDiff
            (data[:, 3] - indicators[:, 4]).reshape(-1, 1),  # Trend (Close - SMA20)
            np.random.uniform(1, 5, len(data)).reshape(-1, 1),  # ATR giả lập
            np.random.uniform(0, 100, len(data)).reshape(-1, 1)  # Stochastic D giả lập
        ])
        
        # Chuẩn hóa dữ liệu cho từng mô hình
        data_lstm = lstm_scaler.transform(data_ml)
        data_cnn = cnn_scaler.transform(data_ml)
        data_tcn = tcn_scaler.transform(data_ml)
        data_xgb = xgb_scaler.transform(data_xgb)
        
        # Định dạng dữ liệu cho từng mô hình
        X_lstm = np.array([data_lstm[-lookback:]])
        X_cnn = np.array([data_cnn[-lookback:]])
        X_cnn = X_cnn[..., np.newaxis]
        X_tcn = np.array([data_tcn[-lookback:]])
        X_tcn = X_tcn[..., np.newaxis]
        X_xgb = np.array([data_xgb[-lookback:].flatten()])
        
        # Xử lý ảnh chart
        chart_image_base64 = data_json["chart_image"]
        img_data = base64.b64decode(chart_image_base64)
        img = Image.open(BytesIO(img_data))
        img = img.resize((64, 64))
        X_image = np.array(img) / 255.0
        X_image = np.expand_dims(X_image, axis=0)
        logging.info("Tiền xử lý dữ liệu thành công")
        
        return df, X_lstm, X_cnn, X_image, X_tcn, X_xgb
    except Exception as e:
        logging.error(f"Lỗi khi tiền xử lý dữ liệu: {e}")
        raise

def analyze_data(data_json):
    """
    Phân tích dữ liệu từ EA để tạo tín hiệu giao dịch.
    
    Parameters:
    - data_json: Dữ liệu JSON từ EA.
    
    Returns:
    - result: Dictionary chứa tín hiệu và confidence {"signal": "...", "confidence": ...}.
    """
    try:
        logging.info("Nhận dữ liệu từ EA: %s", data_json)
        
        # Tiền xử lý dữ liệu
        df, X_lstm, X_cnn, X_image, X_tcn, X_xgb = preprocess_data(data_json)
        
        # Tạo cerebro để chạy chiến lược Backtrader
        cerebro = bt.Cerebro()
        data_feed = CustomDataFeed(dataname=df)
        cerebro.adddata(data_feed)
        logging.info("Khởi tạo Backtrader Cerebro thành công")
        
        # Thêm tất cả chiến lược
        signals = []
        for name, strategy_class in STRATEGIES:
            cerebro.addstrategy(strategy_class)
            logging.info(f"Đã thêm chiến lược: {name}")
        
        # Chạy Backtrader
        cerebro.run()
        logging.info("Chạy Backtrader thành công")
        
        # Thu thập tín hiệu từ các chiến lược
        for name, _ in STRATEGIES:
            strategy = cerebro.strats[0][0]  # Lấy chiến lược đầu tiên
            if hasattr(strategy, 'buy_signal') and strategy.buy_signal[0]:
                signals.append("Buy")
                logging.info(f"{name}: Tín hiệu Buy")
            elif hasattr(strategy, 'sell_signal') and strategy.sell_signal[0]:
                signals.append("Sell")
                logging.info(f"{name}: Tín hiệu Sell")
            elif hasattr(strategy, 'close_signal') and strategy.close_signal[0]:
                signals.append("Sell")
                logging.info(f"{name}: Tín hiệu Sell (Close)")
            else:
                signals.append("None")
                logging.info(f"{name}: Tín hiệu None")
        
        # Thu thập tín hiệu từ mô hình máy học
        ml_signal, ml_confidence = ensemble_model.predict(X_lstm, X_cnn, X_image, X_tcn, X_xgb)
        signals.append(ml_signal)
        logging.info(f"Machine Learning (Ensemble): Tín hiệu {ml_signal}, Confidence: {ml_confidence:.2f}")
        
        # Tổng hợp tín hiệu bằng majority voting với trọng số
        buy_count = signals.count("Buy")
        sell_count = signals.count("Sell")
        none_count = signals.count("None")
        total_strategies = len(signals) - 1  # Không tính ML
        
        # Trọng số: Chiến lược (40%), ML (60%)
        strategy_weight = 0.4 / total_strategies
        ml_weight = 0.6
        
        buy_score = (buy_count * strategy_weight) + (ml_weight if ml_signal == "Buy" else 0)
        sell_score = (sell_count * strategy_weight) + (ml_weight if ml_signal == "Sell" else 0)
        none_score = (none_count * strategy_weight) + (ml_weight if ml_signal == "None" else 0)
        
        # Xác định tín hiệu cuối
        scores = {"Buy": buy_score, "Sell": sell_score, "None": none_score}
        final_signal = max(scores, key=scores.get)
        confidence = scores[final_signal]
        
        logging.info(f"Tín hiệu cuối cùng: {final_signal}, Confidence: {confidence:.2f}")
        return {"signal": final_signal, "confidence": confidence}
    except Exception as e:
        logging.error(f"Lỗi khi phân tích dữ liệu: {e}")
        return {"signal": "None", "confidence": 0.0}

# ZeroMQ server
def start_server():
    """
    Khởi động ZeroMQ server để nhận dữ liệu từ EA và trả về tín hiệu giao dịch.
    """
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        logging.info("ZeroMQ server khởi động tại tcp://*:5555")
        
        while True:
            try:
                # Nhận dữ liệu từ EA
                message = socket.recv_string()
                logging.info(f"Nhận dữ liệu từ EA: {message}")
                
                # Parse JSON
                try:
                    data = json.loads(message)
                except json.JSONDecodeError as e:
                    logging.error(f"Lỗi parse JSON: {e}")
                    socket.send_string(json.dumps({"signal": "None", "confidence": 0.0}))
                    continue
                
                # Phân tích dữ liệu
                result = analyze_data(data)
                
                # Gửi phản hồi
                socket.send_string(json.dumps(result))
                logging.info(f"Gửi phản hồi: {json.dumps(result)}")
                
            except Exception as e:
                logging.error(f"Lỗi xử lý yêu cầu từ EA: {e}")
                socket.send_string(json.dumps({"signal": "None", "confidence": 0.0}))
    except Exception as e:
        logging.error(f"Lỗi khởi động ZeroMQ server: {e}")
        raise

if __name__ == "__main__":
    start_server()