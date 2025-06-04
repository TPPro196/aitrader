import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LayerNormalization, Activation, Input, Dense, Dropout, Add, GlobalAveragePooling2D
import numpy as np

class TCNModel:
    def __init__(self, input_shape=(28, 28, 1), num_classes=3, num_blocks=5, kernel_size=2, dilation_base=2):
        """
        Khởi tạo mô hình Temporal Convolutional Network (TCN).
        
        Parameters:
        - input_shape: Shape của dữ liệu đầu vào (timesteps, features, channels).
        - num_classes: Số lớp đầu ra (3: Buy, Sell, None).
        - num_blocks: Số residual blocks.
        - kernel_size: Kích thước kernel của Conv2D.
        - dilation_base: Hệ số dilation cơ bản (tăng theo lũy thừa).
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.model = self.build_model()

    def residual_block(self, x, filters, dilation_rate):
        """
        Xây dựng một residual block với dilated convolutions.
        
        Parameters:
        - x: Đầu vào của block.
        - filters: Số filters cho Conv2D.
        - dilation_rate: Hệ số dilation.
        
        Returns:
        - Đầu ra của residual block.
        """
        shortcut = x
        x = Conv2D(filters, (self.kernel_size, self.kernel_size), padding='same', dilation_rate=(dilation_rate, 1))(x)
        x = LayerNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        x = Conv2D(filters, (self.kernel_size, self.kernel_size), padding='same', dilation_rate=(dilation_rate, 1))(x)
        x = LayerNormalization()(x)
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
        x = Add()([shortcut, x])
        x = Activation('relu')(x)
        return x

    def build_model(self):
        """
        Xây dựng kiến trúc mô hình TCN.
        
        Returns:
        - model: Mô hình TensorFlow/Keras đã được compile.
        """
        inputs = Input(shape=self.input_shape)
        x = inputs
        
        # Thêm các residual blocks với dilation tăng dần
        filters = 32  # Giảm từ 64 xuống 32
        filter_list = [32, 64, 128, 256, 512]  # Điều chỉnh tăng chậm hơn
        for i in range(self.num_blocks):
            dilation_rate = self.dilation_base ** i
            x = self.residual_block(x, filter_list[i], dilation_rate)
        
        # Global Average Pooling
        x = GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
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
        - X: Dữ liệu đầu vào (numpy array) với shape (samples, timesteps, features, channels).
        
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