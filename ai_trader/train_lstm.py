import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from lstm_model import LSTMModel

# Hàm tính các chỉ báo kỹ thuật
def calculate_technical_indicators(data, timesteps=28):
    """
    Tính 28 chỉ báo kỹ thuật dựa trên dữ liệu OHLC và Volume.
    
    Parameters:
    - data: Dữ liệu OHLC và Volume (numpy array) với shape (samples, 5).
    - timesteps: Số timesteps để tính các chỉ báo.
    
    Returns:
    - Dữ liệu mở rộng với 28 đặc trưng (numpy array) với shape (samples, 28).
    """
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

# Tạo dữ liệu giả lập sát với XAUUSD M5
def generate_dummy_data(num_samples=10000, timesteps=28):
    """
    Tạo dữ liệu giả lập cho XAUUSD khung M5.
    
    Parameters:
    - num_samples: Số lượng mẫu dữ liệu.
    - timesteps: Số timesteps để tính chỉ báo.
    
    Returns:
    - Dữ liệu OHLC và Volume (numpy array).
    """
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
    
    # Tính các chỉ báo kỹ thuật
    indicators = calculate_technical_indicators(data, timesteps)
    
    return data, indicators

# Chuẩn bị dữ liệu cho huấn luyện
def prepare_data(data, indicators, lookback=28):
    """
    Chuẩn bị dữ liệu cho mô hình LSTM.
    
    Parameters:
    - data: Dữ liệu OHLC và Volume (numpy array).
    - indicators: Các chỉ báo kỹ thuật (numpy array).
    - lookback: Số timesteps để tạo sequence.
    
    Returns:
    - X: Dữ liệu đầu vào (numpy array) với shape (samples, lookback, features).
    - y: Nhãn (numpy array) với shape (samples, num_classes).
    - scaler: Đối tượng MinMaxScaler đã được fit.
    """
    scaler = MinMaxScaler()
    data_combined = np.hstack([indicators])
    data_scaled = scaler.fit_transform(data_combined)
    
    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i])
        if data[i, 3] > data[i-1, 3] + 0.5:  # Close tăng đáng kể
            label = 2  # Buy
        elif data[i, 3] < data[i-1, 3] - 0.5:  # Close giảm đáng kể
            label = 0  # Sell
        else:
            label = 1  # None
        y.append(label)
    
    X = np.array(X)
    y = to_categorical(y, num_classes=3)
    return X, y, scaler

# Huấn luyện mô hình
def train_lstm():
    """
    Huấn luyện mô hình LSTM và lưu mô hình cùng scaler.
    """
    # Tạo dữ liệu giả lập
    data, indicators = generate_dummy_data(10000)
    X, y, scaler = prepare_data(data, indicators)
    
    # Chia dữ liệu thành train, validation, test
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
    
    # Khởi tạo và huấn luyện mô hình
    model = LSTMModel(input_shape=(X.shape[1], X.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    checkpoint = ModelCheckpoint("C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\lstm_model.keras", save_best_only=True)
    
    # Huấn luyện mô hình
    history = model.model.fit(X_train, y_train,
                             validation_data=(X_val, y_val),
                             epochs=200,
                             batch_size=64,
                             callbacks=[early_stopping, reduce_lr, checkpoint],
                             verbose=1)
    
    # Đánh giá trên tập test
    test_loss, test_accuracy = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Lưu scaler
    np.save("C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\lstm_scaler.npy", scaler)

if __name__ == "__main__":
    train_lstm()