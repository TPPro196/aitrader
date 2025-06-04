import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import pandas as pd

class XGBoostModel:
    def __init__(self, num_classes=3):
        """
        Khởi tạo mô hình XGBoost.
        
        Parameters:
        - num_classes: Số lớp đầu ra (3: Buy, Sell, None).
        """
        self.model = None
        self.params = {
            'objective': 'multi:softmax',
            'num_class': num_classes,
            'eval_metric': 'mlogloss'
        }

    def optimize_params(self, X, y):
        """
        Tối ưu tham số bằng Bayesian Optimization.
        
        Parameters:
        - X: Dữ liệu đầu vào (numpy array).
        - y: Nhãn (numpy array).
        """
        def xgb_cv(max_depth, learning_rate, n_estimators, gamma, min_child_weight, subsample, colsample_bytree):
            params = {
                'objective': 'multi:softmax',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': int(max_depth),
                'learning_rate': learning_rate,
                'n_estimators': int(n_estimators),
                'gamma': gamma,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree
            }
            model = xgb.XGBClassifier(**params)
            score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
            return score

        pbounds = {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (100, 500),
            'gamma': (0, 5),
            'min_child_weight': (1, 10),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0)
        }
        optimizer = BayesianOptimization(f=xgb_cv, pbounds=pbounds, random_state=1)
        optimizer.maximize(init_points=5, n_iter=15)
        best_params = optimizer.max['params']
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        self.params.update(best_params)
        print("Best parameters:", self.params)

    def fit(self, X, y):
        """
        Huấn luyện mô hình XGBoost.
        
        Parameters:
        - X: Dữ liệu đầu vào (numpy array).
        - y: Nhãn (numpy array).
        """
        self.optimize_params(X, y)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)

    def predict(self, X):
        """
        Dự đoán tín hiệu Buy/Sell/None từ dữ liệu đầu vào.
        
        Parameters:
        - X: Dữ liệu đầu vào (numpy array).
        
        Returns:
        - Dự đoán (numpy array).
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Dự đoán xác suất cho từng lớp.
        
        Parameters:
        - X: Dữ liệu đầu vào (numpy array).
        
        Returns:
        - Xác suất (numpy array) với shape (samples, num_classes).
        """
        return self.model.predict_proba(X)

    def save(self, filepath):
        """
        Lưu mô hình vào file.
        
        Parameters:
        - filepath: Đường dẫn để lưu mô hình (định dạng .json).
        """
        self.model.save_model(filepath)

    def load(self, filepath):
        """
        Load mô hình từ file.
        
        Parameters:
        - filepath: Đường dẫn đến file mô hình (.json).
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(filepath)