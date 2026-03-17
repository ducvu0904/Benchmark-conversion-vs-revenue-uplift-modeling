import lightgbm as lgb
from econml.dml import NonParamDML

def get_rlearner_revenue(final_params, seed):
    """
    Khởi tạo R-Learner cho bài toán Revenue.
    - Y (spend): liên tục -> model_y BẮT BUỘC là Regressor để tránh lỗi target
    - T (treatment): nhị phân -> model_t là Classifier
    - Effect (Uplift): liên tục -> model_final là Regressor
    """
    model_y = lgb.LGBMRegressor(random_state=seed, verbose=-1)
    model_t = lgb.LGBMClassifier(random_state=seed, verbose=-1)
    model_final = lgb.LGBMRegressor(**final_params, random_state=seed, verbose=-1)
    
    return NonParamDML(
        model_y=model_y,
        model_t=model_t,
        model_final=model_final,
        discrete_treatment=True,
        random_state=seed
    )