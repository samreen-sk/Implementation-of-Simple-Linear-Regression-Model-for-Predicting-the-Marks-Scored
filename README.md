# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program and Output:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SHAIK SAMREEN
RegisterNumber:  212223110047
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:\\Users\\admin\\OneDrive\\Desktop\\ML\\DATASET-20250226\\student_scores.csv")
df.head()
```
![417667930-bde309e4-94cd-4622-a7c7-c086b63f6020](https://github.com/user-attachments/assets/ca2a5e2c-3322-4137-94c5-86d8ec2fa760)

```
df.tail()
```
![417668226-73a6fcf1-a680-4253-a1d2-67056da52e45](https://github.com/user-attachments/assets/be15abc0-f6ba-46e4-b605-5ae357f7d9af)
```
x=df.iloc[:,:-1].values
x
```
![417668498-34138173-8cb5-4080-8f34-777d0ab584e8](https://github.com/user-attachments/assets/d80d9ddd-e1ba-41fa-be83-dcf9c6c41f1c)
```
y=df.iloc[:,1].values
y
```
![417668715-fb590fae-d0c0-4460-9b63-26aa32a117c3](https://github.com/user-attachments/assets/bcb495d6-3ee0-48ab-8c46-a5badee3d42f)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)

y_pred
```
![417669044-261601d7-5f68-437b-b2db-044b76d72be0](https://github.com/user-attachments/assets/b4f27e14-47c8-4281-b16b-9933dfd9fde5)
```
y_test
```
![417669195-765bcb23-3d72-4862-82e6-c68cf1ad784d](https://github.com/user-attachments/assets/2b44b4de-7138-4a5c-b879-cebc1ae9a5b3)
```
plt.scatter(x_train, y_train, color="orange")
plt.plot(x_train, reg.predict(x_train), color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
![417669354-6e8a44d4-cf3a-49fc-a99b-032ff502885c](https://github.com/user-attachments/assets/3a43cd58-b93e-479a-8750-460810051f41)
```
plt.scatter(x_test, y_test, color="purple")
plt.plot(x_test, reg.predict(x_test), color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
![417669553-33c3e445-5836-4999-adce-9704890bc78a](https://github.com/user-attachments/assets/2e92877a-948b-4de0-a3ae-caeb07da47dc)
```
mse = mean_squared_error(y_test, y_pred)
print('MSE = ', mse)

mae = mean_absolute_error(y_test, y_pred)
print('MAE = ', mae)

rmse = np.sqrt(mse)
print("RMSE = ", rmse)

```
![417669787-45128e9e-9029-4453-9b8b-8faffeadd5a9](https://github.com/user-attachments/assets/fb43d652-46ff-4b29-84ab-4eee843be0d4)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
