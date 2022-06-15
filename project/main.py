import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression   # 1. Linear Regression 
from sklearn.linear_model import Lasso              # 2. Lasso
from sklearn.linear_model import Ridge              # 3. Ridge
from xgboost.sklearn import XGBRegressor            # 4. XGBoost
from lightgbm.sklearn import LGBMRegressor          # 5. LightGBM
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.load_data import load_data
from src.preprocess import feature_selection, rm_MissingValue
from src.FE import matchType_classify, matchType_encoding, scaling, average_speed, \
    headshotKillsPerc
from src.model import training
 

df_train = load_data('train')

# 1. Preprocessing
## 결측치 처리
train_prep = rm_MissingValue(df_train)

## feature selection
train_prep = feature_selection(train_prep)

# 2. Feature engineering

## Categorical feature encoding
#train_FE = matchType_classify(train_prep)
train_FE = matchType_encoding(train_prep)

## Create new feature
train_FE['average_speed'] = average_speed(df_train)
train_FE['headshotKills'] = df_train.apply(headshotKillsPerc, axis=1)
#train_FE['roadKills'] = df_train.apply(roadKillsPerc, axis=1)

## Normalization(scaling)
train_FE = scaling(train_FE, MinMaxScaler())

# 3. Train
X = train_FE.drop(columns='winPlacePerc')
y = train_FE.winPlacePerc

mae1 = training(LinearRegression(), X, y)
mae2 = training(Lasso(), X, y)
mae3 = training(Ridge(), X, y)
mae4 = training(XGBRegressor(), X, y)
mae5 = training(LGBMRegressor(), X, y)

print("1. Linear Regression : %.4f" % mae1)
print("2. Lasso             : %.4f" % mae2)
print("3. Ridge             : %.4f" % mae3)
print("4. XGBRegressor      : %.4f" % mae4)
print("5. LGBMRegressor     : %.4f" % mae5)

# 4. Test





