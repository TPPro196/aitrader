import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, Input
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.models import Model
import numpy as np

class LSTMModel:
    def __init__(self, input_shape=(28, 28), num_classes=3):
        """
        Khởi tạo mô hình LSTM với Attention Mechanism.
        
        Parameters:
        - input_shape: Tuple (timesteps, features), ví dụ (28, 28) với 28 timesteps và 28 đặc trưng.
        - num_classes: Số lớp đầu ra (3: Buy, Sell, None).
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        """
        Xây dựng kiến trúc mô hình LSTM.
        
        Returns:
        - model: Mô hình TensorFlow/Keras đã được compile.
        """
        inputs = Input(shape=self.input_shape)
        
        # Layer 1: LSTM với 256 units
        x = LSTM(256, return_sequences=True)(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Layer 2: LSTM với 128 units
        x = LSTM(128, return_sequences=True)(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Layer 3: LSTM với 64 units
        x = LSTM(64, return_sequences=True)(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Attention Mechanism
        attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        
        # Layer 4: LSTM với 32 units
        x = LSTM(32, return_sequences=True)(attn_output)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Layer 5: LSTM với 16 units
        x = LSTM(16)(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def predict(self, X):
        """
        Dự đoán tín hiệu Buy/Sell/None từ dữ liệu đầu vào.
        
        Parameters:
        - X: Dữ liệu đầu vào (numpy array) với shape (samples, timesteps, features).
        
        Returns:
        - Dự đoán (numpy array) với shape (samples, num_classes).
        """
        return self.model.predict(X)

    def save(self, filepath):
        """
        Lưu mô hình vào file.
        
        Parameters:
        - filepath: Đường dẫn để lưu mô hình (định dạng .keras).
        """
        self.model.save(filepath)

    def load(self, filepath):
        """
        Load mô hình từ file.
        
        Parameters:
        - filepath: Đường dẫn đến file mô hình (.keras).
        """
        self.model = tf.keras.models.load_model(filepath)