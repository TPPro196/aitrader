import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Add, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np

class CNNModel:
    def __init__(self, input_shape_data=(28, 28, 1), input_shape_image=(64, 64, 3), num_classes=3):
        """
        Khởi tạo mô hình CNN với ResNet-style architecture.
        
        Parameters:
        - input_shape_data: Shape của dữ liệu OHLC (timesteps, features, channels).
        - input_shape_image: Shape của ảnh chart (height, width, channels).
        - num_classes: Số lớp đầu ra (3: Buy, Sell, None).
        """
        self.input_shape_data = input_shape_data
        self.input_shape_image = input_shape_image
        self.num_classes = num_classes
        self.model = self.build_model()

    def residual_block(self, x, filters, kernel_size=7):
        """
        Xây dựng một residual block với Conv2D và BatchNormalization.
        
        Parameters:
        - x: Đầu vào của block.
        - filters: Số filters cho Conv2D.
        - kernel_size: Kích thước kernel.
        
        Returns:
        - Đầu ra của residual block.
        """
        shortcut = x
        x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
        x = Add()([shortcut, x])
        x = tf.keras.layers.ReLU()(x)
        return x

    def build_model(self):
        """
        Xây dựng kiến trúc mô hình CNN.
        
        Returns:
        - model: Mô hình TensorFlow/Keras đã được compile.
        """
        # Branch 1: Dữ liệu OHLC
        input_data = Input(shape=self.input_shape_data)
        x1 = Conv2D(32, (7, 7), padding='same', activation='relu')(input_data)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D((2, 2), padding='same')(x1)
        
        # Thêm các residual blocks
        filters = [64, 128, 256, 512, 512]
        for i, f in enumerate(filters):
            x1 = self.residual_block(x1, f)
            # Chỉ áp dụng MaxPooling cho 3 block đầu tiên
            if i < 3:
                x1 = MaxPooling2D((2, 2), padding='same')(x1)
        
        x1 = GlobalAveragePooling2D()(x1)
        
        # Branch 2: Ảnh chart
        input_image = Input(shape=self.input_shape_image)
        x2 = Conv2D(32, (7, 7), padding='same', activation='relu')(input_image)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D((2, 2), padding='same')(x2)
        
        for i, f in enumerate(filters):
            x2 = self.residual_block(x2, f)
            if i < 3:
                x2 = MaxPooling2D((2, 2), padding='same')(x2)
        
        x2 = GlobalAveragePooling2D()(x2)
        
        # Kết hợp 2 branch
        x = tf.keras.layers.Concatenate()([x1, x2])
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(4, activation='relu')(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model([input_data, input_image], outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def predict(self, X_data, X_image):
        """
        Dự đoán tín hiệu Buy/Sell/None từ dữ liệu đầu vào.
        
        Parameters:
        - X_data: Dữ liệu OHLC (numpy array) với shape (samples, timesteps, features, channels).
        - X_image: Ảnh chart (numpy array) với shape (samples, height, width, channels).
        
        Returns:
        - Dự đoán (numpy array) với shape (samples, num_classes).
        """
        return self.model.predict([X_data, X_image])

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