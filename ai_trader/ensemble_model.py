import numpy as np
from ai_trader.lstm_model import LSTMModel
from ai_trader.cnn_model import CNNModel
from ai_trader.tcn_model import TCNModel
from ai_trader.xgboost_model import XGBoostModel

class EnsembleModel:
    def __init__(self):
        """
        Khởi tạo EnsembleModel kết hợp LSTM, CNN, TCN và XGBoost.
        """
        self.lstm = LSTMModel()
        self.cnn = CNNModel()
        self.tcn = TCNModel()
        self.xgb = XGBoostModel()
        
        # Load các mô hình đã huấn luyện
        self.lstm.load("C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\lstm_model.keras")
        self.cnn.load("C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\cnn_model.keras")
        self.tcn.load("C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\tcn_model.keras")
        self.xgb.load("C:\\Users\\ACER\\Downloads\\ai-trader-main\\ai-trader-main\\models\\xgboost_model.json")

    def predict(self, X_data_lstm, X_data_cnn, X_image, X_data_tcn, X_xgb):
        """
        Dự đoán tín hiệu Buy/Sell/None bằng cách kết hợp các mô hình.
        
        Parameters:
        - X_data_lstm: Dữ liệu đầu vào cho LSTM.
        - X_data_cnn: Dữ liệu OHLC cho CNN.
        - X_image: Ảnh chart cho CNN.
        - X_data_tcn: Dữ liệu đầu vào cho TCN.
        - X_xgb: Dữ liệu đầu vào cho XGBoost.
        
        Returns:
        - signal: Tín hiệu (str): "Buy", "Sell", hoặc "None".
        - confidence: Xác suất của tín hiệu (float).
        """
        lstm_pred = self.lstm.predict(X_data_lstm)
        cnn_pred = self.cnn.predict(X_data_cnn, X_image)
        tcn_pred = self.tcn.predict(X_data_tcn)
        xgb_pred = self.xgb.predict_proba(X_xgb)
        
        # Kết hợp bằng weighted majority voting
        weights = [0.3, 0.3, 0.2, 0.2]  # LSTM: 30%, CNN: 30%, TCN: 20%, XGBoost: 20%
        final_pred = (weights[0] * lstm_pred + weights[1] * cnn_pred + 
                      weights[2] * tcn_pred + weights[3] * xgb_pred)
        signal = np.argmax(final_pred, axis=1)[0]
        confidence = np.max(final_pred)
        
        signal_map = {0: "Sell", 1: "None", 2: "Buy"}
        return signal_map[signal], confidence