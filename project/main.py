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
from src.FE import headshotKillsPerc, scaling

from src.load_data import load_data
from src.preprocess import feature_drop, reduce_mem_usage, rm_MissingValue
from src.model import training
 
#%%
# df = load_data('train')
# df.to_pickle('data/raw/train_V2.pkl')
df_train = pd.read_pickle('data/raw/train_V2.pkl')

#%%
## 1. Preprocessing
# 데이터프레임 메모리 사용량 줄이기
train_prep = reduce_mem_usage(df_train)

# 결측치 처리
train_prep = rm_MissingValue(train_prep)

# feature selection
train_prep = feature_drop(train_prep, ['Id','matchId','groupId','killPlace'])

## 2. Feature engineering
train_FE = train_prep
X = train_FE.drop(columns=['winPlacePerc','matchType'])
X_matchType = train_FE.matchType
y = train_FE.winPlacePerc

# Create new feature
X['headshotKillsPerc'] = df_train.apply(lambda x :headshotKillsPerc(x), axis=1)

# Normalization(scaling)
X = scaling(X, MinMaxScaler(), ['damageDealt','longestKill','walkDistance','swimDistance','rideDistance'])

# Categorical feature encoding

# pd.set_option('display.max_columns', None)
# print(X_FE)

#%%
## 3. Train
mae1 = training(LinearRegression(), X, y)
# mae2 = training(Lasso(), X, y)
# mae3 = training(Ridge(), X, y)
# mae4 = training(XGBRegressor(), X, y)
# mae5 = training(LGBMRegressor(), X, y)
# mae6 = training(RandomForestRegressor(), X, y)
# mae7 = training(GradientBoostingRegressor(), X, y)

print("1. Linear Regression : %.4f" % mae1)
# print("2. Lasso             : %.4f" % mae2)
# print("3. Ridge             : %.4f" % mae3)
# print("4. XGBRegressor      : %.4f" % mae4)
# print("5. LGBMRegressor     : %.4f" % mae5)
# print("6. RFRegressor       : %.4f" % mae6)
# print("7. GBRegressor       : %.4f" % mae7)

# Hyper-parameter tuning

## 4. Test

