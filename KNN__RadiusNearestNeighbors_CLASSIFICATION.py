#!/usr/bin/env python
# coding: utf-8

# For Binary Classification

# In[1]:


############################ For regression: f_regression, mutual_info_regression
############################ For classification: chi2, f_classif, mutual_info_classif
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, mutual_info_regression, mutual_info_classif, chi2
from time import time


# In[2]:


df = pd.read_csv('cancer.csv')


# In[3]:


df.head(100)


# In[4]:



# FEATURES FROM RIDGECV + SELECTFROMMODEL METHOD
df = df[['compactness_mean','concave points_mean','smoothness_se','concavity_se','concave points_se','fractal_dimension_se','fractal_dimension_worst','diagnosis']]    
# FEATURES FROM RIDGECV METHOD
#df = df[['compactness_mean','smoothness_se','concave points_se','fractal_dimension_se','fractal_dimension_worst','concavity_se','diagnosis']]


# In[5]:


df.head(10)


# In[6]:


np.array(df.columns)


# In[7]:


sns.heatmap(df.corr())


# In[8]:


df['diagnosis'].replace(['M','B'], [1,0], inplace = True)
df.head(100)


# In[9]:


df['diagnosis'].value_counts()


# In[10]:


#Y = df.iloc[:, 1].values
Y =  df["diagnosis"].values
print(Y.shape)
print(type(Y))
#print(Y)


# In[11]:


df = df.drop(["diagnosis"], axis=1)

X = df.values
print(X.shape)
print(type(X))
print(X)


# In[12]:


# #CHECKING THE FEATURE IMPORTANCE USING THE SELECTKBEST ALSO KNOWN AS CHI2

X_new = SelectKBest(chi2, k=7).fit_transform(X, Y)
print(X_new.shape)

# #CHECKING THE FEATURE IMPORTANCE USING THE RIDGECV + SELECTFROMMODEL
# ridge = RidgeCV(alphas=np.logspace(-10, 10, num=5)).fit(X, Y)
# importance = np.abs(ridge.coef_)
# feature_names = np.array(df.columns)

# threshold = np.sort(importance)[-8] + 0.01

# tic = time()
# sfm = SelectFromModel(ridge, threshold=threshold).fit(X, Y)
# toc = time()
# print(f"Features selected by SelectFromModel: {feature_names[sfm.get_support()]}")
# print(f"Done in {toc - tic:.3f}s")


# In[13]:


print(X_new)


# In[14]:


Y_new = Y
print(Y_new)


# In[15]:


# IMP = pd.DataFrame(importance) 
# #print(IMP)
# FEATNAMES =  pd.DataFrame(feature_names)
# #print(FEATNAMES)
# frames = [IMP, FEATNAMES]
# result = pd.concat([IMP, FEATNAMES], axis=1)
# (result)

# CONSIDERING FEATURES FROM RIDGECV
# - compactness_mean
# - smoothness_se
# - concave points_se
# - fractal_dimension_se
# - fractal_dimension_worst
# - concavity_se


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_new,Y_new, test_size=0.3, random_state=42)


# In[17]:


X_train[0:5]


# In[18]:


X_test[0:5]


# In[19]:


scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)


# In[20]:


X_train[0:5]


# In[21]:


X_test[0:5]


# In[22]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print(type(X_train))
print(type(Y_train))


# In[23]:


X_train
#type(X_train)


# In[24]:


Y_train
#type(Y_train)


# In[25]:


X_test[0]


# In[26]:


knnCls = RadiusNeighborsClassifier(radius=10.0,p=2, weights='uniform')


# In[27]:


knnCls.fit(X_train,Y_train)


# In[28]:


Y_Pred_TEST = knnCls.predict(X_test)
print(Y_Pred_TEST)
Y_Pred_TRAIN = knnCls.predict(X_train)
#sc =  knnCls.score(Y_test,Y_Pred_TEST)
#print(sc)
#sc2 = knnCls.score(Y_train,Y_Pred_TRAIN)
#print(sc2)

# print(" ACCURACY SCORE FOR TEST DATA : ",lrRegression.score(Y_test, Y_Pred_TEST))
# print(" ACCURACY SCORE FOR TRAIN DATA : ",lrRegression.score(Y_train, Y_Pred_TRAIN))
#Y_Pred = lrRegression.predict(X_test[0].reshape(1,-1))

# for one value prediction
#print(lrRegression.predict(X_test[0].reshape(1,-1)))

# for multiple value prediction
#print(lrRegression.predict(X_test[0:10]))


# In[29]:


result = confusion_matrix(Y_test, Y_Pred_TEST)
print("TEST DATA  Confusion Matrix:")
print(result)


# In[30]:


result = confusion_matrix(Y_train, Y_Pred_TRAIN)
print("TRAIN DATA  Confusion Matrix:")
print(result)


# In[31]:


print(Y_Pred_TEST.shape)
print(Y_Pred_TRAIN.shape)


# In[32]:


acc =  accuracy_score(Y_test, Y_Pred_TEST)
acc2 = accuracy_score(Y_train, Y_Pred_TRAIN)
print("TEST DATA PRedict : ", acc)
print("TRAIN DATA PRedict : ",acc2)


# ----------------------------------------------------------------------------------------------------------
# FROM STEP SELECTKBEST WHEN k = 7 AND K VALUE FOR NEOGHBORS NEAIGBOR = 4
# TEST DATA PRedict :  0.9473684210526315
# TRAIN DATA PRedict :  0.9274725274725275
# ----------------------------------------------------------------------------------------------------------

# In[33]:


from sklearn.metrics import confusion_matrix, r2_score, accuracy_score
cm = confusion_matrix(Y_test, Y_Pred_TEST)
cm


# In[34]:


classreport=  classification_report(Y_test, Y_Pred_TEST)
print(classreport)


# PREVIOUS REPORT FROM SELECTKBEST 
#  precision    recall  f1-score   support
# 
#            0       0.97      1.00      0.99        71
#            1       1.00      0.95      0.98        43
# 
#     accuracy                           0.98       114
#    macro avg       0.99      0.98      0.98       114
# weighted avg       0.98      0.98      0.98       114
# 

# In[35]:


classreport2 =  classification_report(Y_train, Y_Pred_TRAIN)
print(classreport2)


# PREVIOUS REPORT FROM SELECTKBEST
#  precision    recall  f1-score   support
# 
#            0       0.94      0.96      0.95       286
#            1       0.93      0.89      0.91       169
# 
#     accuracy                           0.93       455
#    macro avg       0.93      0.93      0.93       455
# weighted avg       0.93      0.93      0.93       455

# In[36]:


# SCORE ACCURACY FOR TEST DATA
score = knnCls.score(X_test, Y_test)
score


# In[37]:


# SCORE ACCURACY FOR TRAIN DATA
score = knnCls.score(X_train, Y_train)
score


# TEST DATA prev 1 accuracy 
# 0.9824561403508771

# In[38]:


plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cmap='Blues_r', square=True)
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
all_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_title,  size=20)


# In[39]:


knnCls.classes_


# In[40]:


# MSE FOR TEST PREDICTION
mse = mean_squared_error(Y_test, Y_Pred_TEST) 
mse


# In[41]:


# MSE FOR TRAIN PREDICTION
mse = mean_squared_error(Y_train, Y_Pred_TRAIN) 
mse


# In[ ]:




