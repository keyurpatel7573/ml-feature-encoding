import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('cars.csv')
df['brand'].value_counts()

#  OHE using pandas, it is not used generally as it can randomly generate sequance of columns
pd.get_dummies(df,columns=['fuel','owner'],drop_first=True)

X_train,X_test,y_train,y_test = train_test_split(df.ilocloc[:,:4],df.iloc[:,-1],test_size=0.3)

ohe= OneHotEncoder(drop = 'first',sparse_output= False)
X_train_encoder = ohe.fit_transform(X_train[['fuel','owner']])
X_test_encoder = ohe.transform(X_test[['fuel','owner']])

np.hstack((X_train[['brand','km_driven']].values,X_train_encoder))