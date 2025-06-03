import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('heart.csv')    
print(df.head())
scaler = StandardScaler()
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

df[features] = scaler.fit_transform(df[features])

NB = GaussianNB()
x = df.drop(columns=['target'])
y = df['target']
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)


NB.fit(x_train, y_train)


y_pred = NB.predict(x_test)
print('ACCURACY is', accuracy_score(y_test, y_pred))
