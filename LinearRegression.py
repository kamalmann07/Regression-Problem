import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import  cross_validate
import sklearn as sp
from math import sqrt


# import data from csv
data = pd.read_csv('C:/Users/Kamal/Downloads/Assignment Dataset/student_por.csv', header=0)

#prepare data
x_lin= np.array(data.g2).reshape((len(data.g2), 1))
y_lin = np.array(data.g3)

x_train, x_test, y_train, y_test = sp.model_selection.train_test_split(x_lin,y_lin, random_state=1)

#initialize and train the model
lr = LinearRegression()
lr.fit(x_train,y_train)

#calculate linear parameters of the model
print ("coef_ " ,lr.coef_)
print ("intercept_ " ,lr.intercept_)
y_pred = lr.predict(x_test)

print("Root mean squared error ",sqrt(sp.metrics.mean_squared_error(y_test,y_pred)))
print ("R2 Score is", sp.metrics.r2_score(y_test,y_pred))

#cross validation with 5 folds
scores = sp.model_selection.cross_val_score(lr, x_train,y_train, cv=5, scoring= 'neg_mean_squared_error')
print ("Root mean squared error for cross validation is ",sqrt(-(scores.mean())))


# scoreMul = cross_validate(lr, x_lin,lr.predict(x_lin), scoring=('r2', 'neg_mean_squared_error'), cv=3,return_train_score=True)

pyplot.scatter(x_lin,y_lin,color='black',label='Initial Data')
pyplot.plot(x_lin,lr.predict(x_lin),color='blue',label='Linear Regression Model')
pyplot.xlabel('g2')
pyplot.ylabel('g3')
pyplot.title('Linear Regression')
pyplot.legend()
pyplot.show()