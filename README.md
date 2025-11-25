# 11-Implementation-of-SVM-For-Spam-Mail-Detection

Name : SANJAY S

Register Number: 212223040184

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.
   
2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: SANJAY S
RegisterNumber: 212223040184

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data
data.info
data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
DATA:

<img width="917" height="502" alt="328882696-5d1f74f3-8c4b-4fa4-b393-1b43b1c65733" src="https://github.com/user-attachments/assets/0a0254c8-e168-4fa9-9a5e-c0c63ce9aa6f" />

Confusion matrix:

<img width="125" height="48" alt="328883151-98ae0d93-d355-43f3-b59a-52bb58e13d5e" src="https://github.com/user-attachments/assets/6a534a38-30a3-42b4-883d-139e6d8d6d2e" />

classification:

<img width="602" height="221" alt="Screenshot 2025-10-29 205843" src="https://github.com/user-attachments/assets/04d05fbe-6c6c-4b6e-a7ab-2e9f31d5b0f9" />

Accuracy:

<img width="237" height="38" alt="328883112-d1d69362-7f5b-4249-a95b-e9f5e894a917" src="https://github.com/user-attachments/assets/cabbb73e-1900-467c-b27b-9d49cdc945d0" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
