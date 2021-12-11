import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA

df = pd.read_csv('C:\\Users\\vishe\\Desktop\\ML\\Iris.csv') # reading the dataset

df = df.iloc[:,1:]  #Removing Id column

X = df.iloc[:, :-1] # Storing features

Y = df.loc[:,"Species"]    #Storing output

def mappingClass(s):  #This function for maps output values to integer
    r = 0
    if(s == 'Iris-setosa'):
        r = 1
    return r

Y_map = np.array(list(map(mappingClass, Y)))  #Calling map function to convert output into integer

sc = MinMaxScaler()         #creating minmax scalar object

X = sc.fit_transform(X)    #normalizing and transforming data

X = PCA(n_components=2).fit_transform(X)   #performing PCA for reducing the number of features to 2.

# splitting the data set to training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(       
    X,
    Y_map,
    test_size=0.3,
    random_state=52
)

sv_c = SVC(kernel="linear")  #creating a support vector classifier object

sv_c.fit(X_train, Y_train)   # training the model with training data set

plot_decision_regions(X = X_train, y = Y_train, clf = sv_c, legend = 2)  #plotting graph
plt.show()

Y_pred = sv_c.predict(X_test)    #making predictions

print("Accuracy: ",sv_c.score(X_test, Y_test)*100, "%")   #printing the accuracy of classifier