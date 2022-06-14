#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[32]:


base_path = '../data/raw/'

df_train = pd.read_csv(base_path + 'train_V2.csv')
df_test = pd.read_csv(base_path + 'test_V2.csv')
submission = pd.read_csv(base_path + 'sample_submission_V2.csv')


# # Preprocessing

# In[34]:


# 결측치 확인
df_train[df_train.isnull().any(axis=1)]


# In[53]:


# 결측치 제거
train = df_train.dropna(axis=0)
train.info()


# In[54]:


# 학습에 사용할 컬럼을 추출
train = train.drop(columns=['Id','killPlace','killPoints','matchDuration',    'matchId','rankPoints','teamKills','winPoints','groupId','numGroups',        'maxPlace'])


# In[57]:


train.head()


# # Feature Engineering

# ## matchType Ordinal encoding

# In[59]:


# custom match
train.loc[(train.matchType.str.contains('normal'))|    (train.matchType.str.contains('flare'))|        (train.matchType.str.contains('crash')), 'matchType'] = 'custom'


# In[60]:


# standard match
train.loc[train.matchType.str.contains('solo'), 'matchType'] = 'solo'
train.loc[train.matchType.str.contains('duo'), 'matchType'] = 'duo'
train.loc[train.matchType.str.contains('squad'), 'matchType'] = 'squad'


# In[106]:


# Nominal Encoding
train_OHE = pd.get_dummies(train, columns=['matchType'])
train_OHE.head(10)


# ## feature scaling

# In[112]:


X = train_OHE.drop(columns='winPlacePerc')
y = train_OHE.winPlacePerc

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
temp = scaler.fit_transform(X.loc[:,:'weaponsAcquired'])
temp


# In[113]:


pd.DataFrame(temp)


# In[121]:


X.loc[:,:'weaponsAcquired'] = temp[:, :]


# In[135]:


X.columns


# In[136]:


train_OHE.columns


# In[134]:


y


# # Training

# In[124]:


# 학습을 위한 라이브러리 세팅
from sklearn.linear_model import LinearRegression   # 1. Linear Regression 
from sklearn.linear_model import Lasso              # 2. Lasso
from sklearn.linear_model import Ridge              # 3. Ridge
from xgboost.sklearn import XGBRegressor            # 4. XGBoost
from lightgbm.sklearn import LGBMRegressor          # 5. LightGBM

# 평가 지표
from sklearn.metrics import mean_absolute_error


# In[132]:


def training(m, t, target):
    model = m
    model.fit(t, target)
    pred_train = model.predict(t)
    mae_train = mean_absolute_error(target, pred_train)
    return mae_train


# In[133]:


print(" 1. Linear Regression : %.4f" % training(LinearRegression(), X, y))
print(" 2. Lasso             : %.4f" % training(Lasso(), X, y))
print(" 3. Ridge             : %.4f" % training(Ridge(), X, y))
print(" 4. XGBoost           : %.4f" % training(XGBRegressor(), X, y))
print(" 5. LigthGBM          : %.4f" % training(LGBMRegressor(), X, y))


# In[ ]:


# Hyper-parameter tuning

# GridSearchCV
from sklearn.model_selection import GridSearchCV

parma_grid = {
    "max_depth" : [],
    "learning_rate" : [],
    "n_estimators" : [],
}


# # Test  
# training set과 같은 전처리를 해줘야 함.

# In[139]:


test = df_test.copy()


# In[140]:


# 결측치 확인
df_test[df_test.isnull().any(axis=1)]


# In[141]:


# 사용할 컬럼만 추출
test = test.drop(columns=['Id','killPlace','killPoints','matchDuration',    'matchId','rankPoints','teamKills','winPoints','groupId','numGroups',        'maxPlace'])


# In[142]:


# custom match
test.loc[(test.matchType.str.contains('normal'))|    (test.matchType.str.contains('flare'))|        (test.matchType.str.contains('crash')), 'matchType'] = 'custom'

# standard match
test.loc[test.matchType.str.contains('solo'), 'matchType'] = 'solo'
test.loc[test.matchType.str.contains('duo'), 'matchType'] = 'duo'
test.loc[test.matchType.str.contains('squad'), 'matchType'] = 'squad'


# In[143]:


test


# In[144]:


# Ordinal Encoding
test_OHE = pd.get_dummies(test, columns=['matchType'])
test_OHE.head(10)


# In[147]:


X_test = test_OHE.copy()


# In[148]:


# feature scaling
scaler = MinMaxScaler()
temp2 = scaler.fit_transform(X_test.loc[:,:'weaponsAcquired'])
pd.DataFrame(temp2)


# In[150]:


X_test.columns


# In[151]:


X_test.loc[:,:'weaponsAcquired'] = temp2[:, :]


# In[154]:


reg = XGBRegressor()
reg.fit(X, y)
result = reg.predict(X_test)
print(result)


# In[156]:


submission['winPlacePerc'] = result
submission.to_csv('submission.csv', index=False)

