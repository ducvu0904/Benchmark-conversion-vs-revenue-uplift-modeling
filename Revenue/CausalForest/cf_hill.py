# cf_hill.py
import numpy as np
from econml.dml import CausalForestDML
from lightgbm import LGBMClassifier, LGBMRegressor

class CausalForestWrapper:
    """
    Wrapper cho CausalForestDML từ thư viện econml.
    Tự động điều chỉnh first-stage model dựa trên task_type (conversion hoặc revenue).
    """
    def __init__(self, task_type='conversion', random_state=42, **cf_params):
        self.task_type = task_type
        self.random_state = random_state
        
        # Mô hình dự đoán Treatment (T) luôn là Classifier
        model_t = LGBMClassifier(random_state=random_state, verbose=-1)
        
        # Thiết lập Outcome model và cấu hình của CausalForestDML
        if self.task_type == 'conversion':
            # Target là phân loại -> Classifier
            model_y = LGBMClassifier(random_state=random_state, verbose=-1)
            
            # Khởi tạo mô hình
            self.model = CausalForestDML(
                model_y=model_y,
                model_t=model_t,
                discrete_treatment=True,
                discrete_outcome=True, # QUAN TRỌNG: Báo cho model biết outcome là rời rạc
                random_state=random_state,
                cv=3,
                **cf_params
            )
            
        elif self.task_type == 'revenue':
            # Target là liên tục -> Regressor
            model_y = LGBMRegressor(random_state=random_state, verbose=-1)
            
            # Khởi tạo mô hình
            self.model = CausalForestDML(
                model_y=model_y,
                model_t=model_t,
                discrete_treatment=True,
                discrete_outcome=False, # Outcome là liên tục
                random_state=random_state,
                cv=3,
                **cf_params
            )
        else:
            raise ValueError("task_type phải là 'conversion' hoặc 'revenue'")

    def fit(self, X, y, treatment):
        """Huấn luyện mô hình Causal Forest"""
        
        # Lớp bảo vệ: Đảm bảo dữ liệu đúng định dạng mong đợi của scikit-learn
        y_data = np.array(y).flatten()
        t_data = np.array(treatment).flatten()
        
        if self.task_type == 'conversion':
            # Ép chặt y_data và t_data thành số nguyên để tránh lỗi "continuous target"
            y_data = y_data.astype(int)
            t_data = t_data.astype(int)
        elif self.task_type == 'revenue':
            # t_data vẫn phải là số nguyên (vì discrete_treatment=True)
            t_data = t_data.astype(int)
            # y_data để dạng số thực bình thường
            
        # Truyền dữ liệu đã định dạng vào mô hình
        self.model.fit(Y=y_data, T=t_data, X=X)

    def predict(self, X):
        """Dự đoán CATE (Uplift Score)"""
        return self.model.effect(X).flatten()