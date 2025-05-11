from pandas._config import config
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
import yaml
import sys
import os
sys.path.append(os.getcwd())

from src.loggers import get_logger

def load_config(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config

def clear_data(path2data):
    df = pd.read_csv(path2data)
    df.describe()

    # нет категориальных данных, чистим только странные выбросы в числовых
    question_Hours = df[(df.Operating_Hours_Per_Day > 16)]
    df = df.drop(question_Hours.index)
    
    question_Revenue = df[df.Daily_Revenue < 0]
    df = df.drop(question_Revenue.index)
    
    question_Employees = df[df.Number_of_Employees < 1]
    df = df.drop(question_Employees.index)

    return df

def scale_frame(frame):
    df = frame.copy()
    X, Y = df.drop(columns = ['Daily_Revenue']), df['Daily_Revenue']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(Y.values.reshape(-1,1))
    return X_scale, Y_scale, power_trans

def featurize(dframe, config) -> None:
    # Генерация новых признаков
    logger = get_logger('FEATURIZE')
    logger.info('Create features')

    dframe['Average_Number_of_Customers_Per_Hour'] = dframe['Number_of_Customers_Per_Day']/ dframe['Operating_Hours_Per_Day']
    dframe['Foot_Traffic_Customers_Ratio'] = dframe['Number_of_Customers_Per_Day']/ dframe['Location_Foot_Traffic']

    features_path = config['featurize']['features_path']
    dframe.to_csv(features_path, index=False)


if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    df_prep = clear_data(config['data_load']['dataset_csv'])
    df_new_featur = featurize(df_prep, config)
    