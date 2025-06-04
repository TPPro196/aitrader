import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost_model import XGBoostModel

# Hàm tính các chỉ báo kỹ thuật (tương tự như trong train_lstm.py)
def calculate_technical_indicators(data, timesteps=28):
    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    df['Yield'] = (df['Close'] - df['Open']) / df['Open']
    df['PercVol'] = df['Volume'].pct_change().fillna(0)
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(lambda x: np.sum(x * weights) / weights.sum(), raw=True)
    
    df['WMA5'] = wma(df['Close'], 5)
    df['WMA10'] = wma(df['Close'], 10)
    df['WMA20'] = wma(df['Close'], 20)
    df['WMA50'] = wma(df['Close'], 50)
    
    def hma(series, period):
        wma_half = wma(series, period // 2)
        wma_full = wma(series, period)
        raw_hma = 2 * wma_half - wma_full
        return wma(raw_hma, int(np.sqrt(period)))
    
    df['HMA5'] = hma(df['Close'], 5)
    df['HMA10'] = hma(df['Close'], 10)
    df['HMA20'] = hma(df['Close'], 20)
    df['HMA50'] = hma(df['Close'], 50)
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mean_dev = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['CCI'] = (typical_price - sma_tp) / (0.015 * mean_dev)
    
    lowest_low = df['Low'].rolling(window=14).min()
    highest_high = df['High'].rolling(window=14).max()
    df['StochOsc'] = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['PPO'] = ((ema12 - ema26) / ema26) * 100
    
    roc1 = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    roc2 = ((df['Close'] - df['Close'].shift(15)) / df['Close'].shift(15)) * 100
    roc3 = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)) * 100
    roc4 = ((df['Close'] - df['Close'].shift(30)) / df['Close'].shift(30)) * 100
    df['KST'] = (roc1 * 1) + (roc2 * 2) + (roc3 * 3) + (roc4 * 4)
    
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BOLM'] = sma20
    df['BOLU'] = sma20 + (2 * std20)
    df['BOLD'] = sma20 - (2 * std20)
    
    # Thêm các đặc trưng mở rộng
    df['PriceDiff'] = df['Close'] - df['Open']
    df['Trend'] = df['Close'] - df['SMA20']
    df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
    df['StochD'] = df['StochOsc'].rolling(window=3).mean()
    
    df = df.fillna(0)
    
    return df[['Yield', 'PercVol', 'SMA5', 'SMA10', 'SMA20', 'SMA50', 'EMA5', 'EMA10', 'EMA20', 'EMA50',
               'WMA5', 'WMA10', 'WMA20', 'WMA50', 'HMA5', 'HMA10', 'HMA20', 'HMA50', 'MACD', 'CCI',
               'StochOsc', 'RSI', 'ROC', 'PPO', 'KST', 'BOLM', 'BOLU', 'BOLD', 'PriceDiff', 'Trend', 'ATR', 'StochD']].values

# Tạo dữ liệu giả lập
def generate_dummy_data(num_samples=10000, timesteps=28):
    data = []
    price = 2500
    for i in range(num_samples):
        open_price = price + np.random.uniform(-1, 1)
        high = open_price + np.random.uniform(0, 2)
        low = open_price - np.random.uniform(0, 2)
        close = np.random.uniform(low, high)
        volume = np.random.randint(1000, 10000)
        data.append([open_price, high, low, close, volume])
        price = close
    data = np.array(data)
    indicators = calculate_technical_indicators(data, timesteps)
    return data, indicators

# Chuẩn bị dữ liệu
def prepare_data(data, indicators, lookback=28):
    scaler = MinMaxScaler()
    data_combined = np.hstack([indicators])
    data_scaled = scaler.fit_transform(data_combined)
    
    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i].flatten())
        if data[i, 3] > data[i-1, 3] + 0.5:
            label = 2  # Buy
        elif data[i, 3] < data[i-1, 3] - 0.5:
            label = 0  # Sell
        else:
            label = 1  # None
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

# Huấn luyện mô hình
def train_xgboost():
    data, indicators = generate_dummy_data(10000)
    X, y, scaler = prepare_data(data, indicators)
    
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    model = XGBoostModel()
    model.fit(X_train, y_train)
    
    model.save("C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\xgboost_model.json")
    np.save("C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\xgboost_scaler.npy", scaler)

if __name__ == "__main__":
    train_xgboost()