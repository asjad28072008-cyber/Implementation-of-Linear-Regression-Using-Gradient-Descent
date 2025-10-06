# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1. Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Muhammad Asjad E
RegisterNumber:  25013957
*/

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000): 
  X = np.c_[np.ones(len(X1)),X1] 
  theta = np.zeros(X.shape[1]).reshape(-1,1) 
  for _ in range(num_iters): 
    predictions = (X).dot(theta).reshape(-1,1) 
    errors=(predictions - y ).reshape(-1,1) 
    theta -= learning_rate*(1/len(X1))*X.T.dot(errors) 

  return theta 
data=pd.read_csv("50_Startups.csv") 
print(data.head()) 
print("\n") 
X=(data.iloc[1:,:-2].values) 
X1=X.astype(float) 
scaler=StandardScaler() 
y=(data.iloc[1:,-1].values).reshape(-1,1) 
X1_Scaled=scaler.fit_transform(X1) 
Y1_Scaled=scaler.fit_transform(y) 
print(X) 
print("\n") 
print(X1_Scaled) 
print("\n") 
theta= linear_regression(X1_Scaled,Y1_Scaled) 
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1, 1) 
new_Scaled=scaler.fit_transform(new_data) 
prediction=np.dot(np.append(1,new_Scaled),theta) 
prediction=prediction.reshape(-1,1) 
pre=scaler.inverse_transform(prediction) 
print(prediction) 
print(f"Predicted value: {pre}")


```

## Output:

Data Information

<img width="735" height="136" alt="Screenshot 2025-10-06 093919" src="https://github.com/user-attachments/assets/42e1e3ee-893b-4584-b026-92c234058405" />

Value of X


<img width="153" height="408" alt="Screenshot 2025-10-06 093959" src="https://github.com/user-attachments/assets/ce4da4a3-d9a7-4115-9ed8-c50aff3b6f43" />

Value of X1_scaled
<img width="201" height="410" alt="Screenshot 2025-10-06 094604" src="https://github.com/user-attachments/assets/dab7afe0-c9d3-448c-877f-91befe5261fd" />


predicted value

<img width="510" height="73" alt="Screenshot 2025-10-06 094130" src="https://github.com/user-attachments/assets/ad378919-f4e8-4a2d-95b9-66514fed55be" />






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
