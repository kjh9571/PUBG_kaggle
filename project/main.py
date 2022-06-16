#%%
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression   # 1. Linear Regression 
from sklearn.linear_model import Lasso              # 2. Lasso
from sklearn.linear_model import Ridge              # 3. Ridge
from xgboost.sklearn import XGBRegressor            # 4. XGBoost
from lightgbm.sklearn import LGBMRegressor          # 5. LightGBM
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from src.load_data import load_data
from src.preprocess import feature_selection, reduce_mem_usage, rm_MissingValue
from src.FE import matchType_classify, matchType_encoding, scaling, average_speed, total_Distance
from src.model import training
 
#%%
df = load_data('train')
df.to_pickle('data/raw/train_V2.pkl')
df_train = pd.read_pickle('data/raw/train_V2.pkl')

#%%
## 1. Preprocessing
# 데이터프레임 메모리 사용량 줄이기
train_prep = reduce_mem_usage(df_train)

# 결측치 처리
train_prep = rm_MissingValue(train_prep)

# feature selection
train_prep = feature_selection(train_prep)

## 2. Feature engineering
train_FE = train_prep
X = train_FE.drop(columns=['winPlacePerc','matchType'])
X_matchType = train_FE.matchType
y = train_FE.winPlacePerc

# Create new feature
X['average_speed'] = average_speed(df_train)
#X['headshotKillsPerc'] = df_train.apply(headshotKillsPerc, axis=1)
X['totalDistance'] = total_Distance(df_train)

# Normalization(scaling)
X_scaled = scaling(X, MinMaxScaler())

# Categorical feature encoding
X = pd.concat([X_scaled, X_matchType], axis=1)
# X_FE = matchType_classify(X)
# X_FE = matchType_encoding(X_FE)

# pd.set_option('display.max_columns', None)
# print(X_FE)

#%%
## 3. Train
# mae1, reg1 = training(LinearRegression(), X, y)
# mae2, reg2 = training(Lasso(), X, y)
# mae3, reg3 = training(Ridge(), X, y)
mae4, reg4 = training(XGBRegressor(), X, y)
mae5, reg5 = training(LGBMRegressor(), X, y)
mae6, reg6 = training(RandomForestRegressor(), X, y)
mae7, reg7 = training(GradientBoostingRegressor(), X, y)

# print("1. Linear Regression : %.4f" % mae1)
# print("2. Lasso             : %.4f" % mae2)
# print("3. Ridge             : %.4f" % mae3)
print("4. XGBRegressor      : %.4f" % mae4)
print("5. LGBMRegressor     : %.4f" % mae5)
print("6. RFRegressor       : %.4f" % mae6)
print("7. GBRegressor       : %.4f" % mae7)

# Hyper-parameter tuning

## 4. Test

