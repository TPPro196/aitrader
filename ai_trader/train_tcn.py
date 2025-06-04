import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tcn_model import TCNModel

# Hàm tính các chỉ báo kỹ thuật (tối ưu bằng NumPy)
def calculate_technical_indicators(data, timesteps=28):
    """
    Tính 28 chỉ báo kỹ thuật dựa trên dữ liệu OHLC và Volume, sử dụng NumPy thay vì Pandas.
    
    Parameters:
    - data: Dữ liệu OHLC và Volume (numpy array) với shape (samples, 5).
    - timesteps: Số timesteps để tính các chỉ báo.
    
    Returns:
    - Dữ liệu mở rộng với 28 đặc trưng (numpy array) với shape (samples, 28).
    """
    # Trích xuất các cột
    open_price = data[:, 0]
    high = data[:, 1]
    low = data[:, 2]
    close = data[:, 3]
    volume = data[:, 4]
    n = len(close)
    
    # Khởi tạo mảng để lưu các chỉ báo
    indicators = np.zeros((n, 28))
    
    # 1. Yield
    indicators[:, 0] = (close - open_price) / open_price
    
    # 2. Percentage Volume
    indicators[1:, 1] = (volume[1:] - volume[:-1]) / volume[:-1]
    
    # 3-6. Simple Moving Averages (SMA)
    for i in range(n):
        if i >= 4:
            indicators[i, 2] = np.mean(close[i-4:i+1])  # SMA5
        if i >= 9:
            indicators[i, 3] = np.mean(close[i-9:i+1])  # SMA10
        if i >= 19:
            indicators[i, 4] = np.mean(close[i-19:i+1])  # SMA20
        if i >= 49:
            indicators[i, 5] = np.mean(close[i-49:i+1])  # SMA50
    
    # 7-10. Exponential Moving Averages (EMA)
    def ema(data, span):
        ema_values = np.zeros_like(data)
        ema_values[0] = data[0]
        alpha = 2 / (span + 1)
        for i in range(1, len(data)):
            ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
        return ema_values
    
    indicators[:, 6] = ema(close, 5)   # EMA5
    indicators[:, 7] = ema(close, 10)  # EMA10
    indicators[:, 8] = ema(close, 20)  # EMA20
    indicators[:, 9] = ema(close, 50)  # EMA50
    
    # 11-14. Weighted Moving Averages (WMA)
    def wma(data, window):
        wma_values = np.zeros_like(data)
        weights = np.arange(1, window + 1)
        weight_sum = np.sum(weights)
        for i in range(window-1, len(data)):
            wma_values[i] = np.sum(data[i-window+1:i+1] * weights) / weight_sum
        return wma_values
    
    indicators[:, 10] = wma(close, 5)   # WMA5
    indicators[:, 11] = wma(close, 10)  # WMA10
    indicators[:, 12] = wma(close, 20)  # WMA20
    indicators[:, 13] = wma(close, 50)  # WMA50
    
    # 15-18. Hull Moving Averages (HMA)
    def hma(data, period):
        wma_half = wma(data, period // 2)
        wma_full = wma(data, period)
        raw_hma = 2 * wma_half - wma_full
        return wma(raw_hma, int(np.sqrt(period)))
    
    indicators[:, 14] = hma(close, 5)   # HMA5
    indicators[:, 15] = hma(close, 10)  # HMA10
    indicators[:, 16] = hma(close, 20)  # HMA20
    indicators[:, 17] = hma(close, 50)  # HMA50
    
    # 19. MACD
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    indicators[:, 18] = ema12 - ema26
    
    # 20. Commodity Channel Index (CCI)
    typical_price = (high + low + close) / 3
    sma_tp = np.zeros_like(typical_price)
    for i in range(19, n):
        sma_tp[i] = np.mean(typical_price[i-19:i+1])
    mean_dev = np.zeros_like(typical_price)
    for i in range(19, n):
        mean_dev[i] = np.mean(np.abs(typical_price[i-19:i+1] - np.mean(typical_price[i-19:i+1])))
    indicators[:, 19] = (typical_price - sma_tp) / (0.015 * mean_dev + 1e-10)
    
    # 21. Stochastic Oscillator (%K)
    lowest_low = np.zeros_like(close)
    highest_high = np.zeros_like(close)
    for i in range(13, n):
        lowest_low[i] = np.min(low[i-13:i+1])
        highest_high[i] = np.max(high[i-13:i+1])
    indicators[:, 20] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    
    # 22. Relative Strength Index (RSI)
    delta = np.diff(close)
    delta = np.concatenate([np.array([0]), delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(14, n):
        avg_gain[i] = np.mean(gain[i-13:i+1])
        avg_loss[i] = np.mean(loss[i-13:i+1])
    rs = avg_gain / (avg_loss + 1e-10)
    indicators[:, 21] = 100 - (100 / (1 + rs))
    
    # 23. Rate of Change (ROC)
    for i in range(12, n):
        indicators[i, 22] = ((close[i] - close[i-12]) / close[i-12]) * 100
    
    # 24. Percentage Price Oscillator (PPO)
    indicators[:, 23] = ((ema12 - ema26) / (ema26 + 1e-10)) * 100
    
    # 25. Know Sure Thing (KST)
    roc1 = np.zeros_like(close)
    roc2 = np.zeros_like(close)
    roc3 = np.zeros_like(close)
    roc4 = np.zeros_like(close)
    for i in range(30, n):
        roc1[i] = ((close[i] - close[i-10]) / close[i-10]) * 100
        roc2[i] = ((close[i] - close[i-15]) / close[i-15]) * 100
        roc3[i] = ((close[i] - close[i-20]) / close[i-20]) * 100
        roc4[i] = ((close[i] - close[i-30]) / close[i-30]) * 100
    indicators[:, 24] = (roc1 * 1) + (roc2 * 2) + (roc3 * 3) + (roc4 * 4)
    
    # 26-28. Bollinger Bands
    sma20 = np.zeros_like(close)
    std20 = np.zeros_like(close)
    for i in range(19, n):
        sma20[i] = np.mean(close[i-19:i+1])
        std20[i] = np.std(close[i-19:i+1])
    indicators[:, 25] = sma20  # BOLM
    indicators[:, 26] = sma20 + (2 * std20)  # BOLU
    indicators[:, 27] = sma20 - (2 * std20)  # BOLD
    
    # Điền giá trị NaN
    indicators = np.nan_to_num(indicators, 0)
    
    return indicators

# Tạo dữ liệu giả lập
def generate_dummy_data(num_samples=5000, timesteps=28):  # Giảm từ 10000 xuống 5000
    data = np.zeros((num_samples, 5))
    price = 2500
    for i in range(num_samples):
        open_price = price + np.random.uniform(-1, 1)
        high = open_price + np.random.uniform(0, 2)
        low = open_price - np.random.uniform(0, 2)
        close = np.random.uniform(low, high)
        volume = np.random.randint(1000, 10000)
        data[i] = [open_price, high, low, close, volume]
        price = close
    indicators = calculate_technical_indicators(data, timesteps)
    return data, indicators

# Chuẩn bị dữ liệu
def prepare_data(data, indicators, lookback=28):
    scaler = MinMaxScaler()
    data_combined = np.hstack([indicators])
    data_scaled = scaler.fit_transform(data_combined)
    
    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i])
        if data[i, 3] > data[i-1, 3] + 0.5:
            label = 2  # Buy
        elif data[i, 3] < data[i-1, 3] - 0.5:
            label = 0  # Sell
        else:
            label = 1  # None
        y.append(label)
    
    X = np.array(X)
    X = X[..., np.newaxis]
    y = to_categorical(y, num_classes=3)
    return X, y, scaler

# Huấn luyện mô hình
def train_tcn():
    data, indicators = generate_dummy_data(5000)  # Giảm từ 10000 xuống 5000
    X, y, scaler = prepare_data(data, indicators)
    
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
    
    model = TCNModel(input_shape=(X.shape[1], X.shape[2], 1))
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    checkpoint = ModelCheckpoint("C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\tcn_model.keras", save_best_only=True)
    
    model.model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,  # Giảm từ 200 xuống 50
                    batch_size=64,  # Tăng từ 32 lên 64
                    callbacks=[early_stopping, reduce_lr, checkpoint],
                    verbose=1)
    
    test_loss, test_accuracy = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    np.save("C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\tcn_scaler.npy", scaler)

if __name__ == "__main__":
    train_tcn()