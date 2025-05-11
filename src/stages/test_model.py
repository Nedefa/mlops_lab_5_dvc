import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
import mlflow
from prepare_dataset import load_config

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def test_model(config):
    model_path = config['train']['model_path']
    model = joblib.load(model_path)

    power_path = config['train']['power_path']
    power_trans = joblib.load(power_path)

    test_path = config['data_split']['testset_path']
    df_test = pd.read_csv(test_path)
    X_test, y_test = df_test.drop(columns=['Daily_Revenue']).values, df_test['Daily_Revenue'].values

    y_pred = model.predict(X_test)

    y_pred_original = power_trans.inverse_transform(y_pred.reshape(-1, 1))
    y_test_original = power_trans.inverse_transform(y_test.reshape(-1, 1))

    rmse, mae, r2 = eval_metrics(y_test_original, y_pred_original)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    mlflow.set_experiment("linear model revenue")
    with mlflow.start_run():
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("test_mae", mae)
        
    metrics = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
    }
    os.makedirs("dvclive", exist_ok=True) 
    with open("dvclive/metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    test_model(config)
