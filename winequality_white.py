import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('winequality_white.csv') # reading the dataset

df.tail() # first 5 rows

df.info() # to check for null values

df.describe() # to check for features

# selecting the feature
x=df.drop("quality",axis=1)
x

# selecting the target
y=df['quality']
y

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=47
)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

svm=SVC()
svm.fit(x_train, y_train)

pred_svc=svm.predict(x_test)

print(classification_report(y_test, pred_svc))