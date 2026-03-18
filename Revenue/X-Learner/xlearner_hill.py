# File: xlearner_hill.py
from econml.metalearners import XLearner
from lightgbm import LGBMClassifier, LGBMRegressor

def get_xlearner(task_type, params):
    """
    Hàm khởi tạo X-Learner từ thư viện EconML.
    
    Parameters:
    -----------
    task_type : str
        'conversion' (Classification) hoặc 'revenue' (Regression)
    params : dict
        Các tham số Hyperparameters tốt nhất từ Optuna cho LightGBM
        
    Returns:
    --------
    XLearner object
    """
    # 1. First-stage models: Dự đoán Target (Y) dựa trên (X) cho từng nhóm T=0 và T=1
    if task_type == 'conversion':
        # Đối với bài toán Conversion, Target là Binary (0/1) -> Dùng Classifier
        models = LGBMClassifier(**params)
    elif task_type == 'revenue':
        # Đối với bài toán Revenue, Target là Continuous (Spend) -> Dùng Regressor
        models = LGBMRegressor(**params)
    else:
        raise ValueError("task_type phải là 'conversion' hoặc 'revenue'")

    # 2. Propensity model: Dự đoán xác suất nhận Treatment (T) dựa trên (X)
    # Vì T luôn là Binary (0/1), Propensity model bắt buộc phải là Classifier
    propensity_model = LGBMClassifier(**params)

    # 3. CATE (Uplift) models: Dự đoán hiệu ứng tác động (Treatment Effect)
    # LƯU Ý QUAN TRỌNG: Dù bài toán là Conversion hay Revenue, bản thân giá trị 
    # Uplift Score (CATE) luôn là một biến liên tục (Continuous). 
    # Do đó, CATE models BẮT BUỘC phải là Regressor để tránh lỗi RuntimeError/AttributeError.
    cate_models = LGBMRegressor(**params)

    # Khởi tạo XLearner
    x_learner = XLearner(
        models=models,
        cate_models=cate_models,
        propensity_model=propensity_model
    )
    
    return x_learner