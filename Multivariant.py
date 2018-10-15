import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import sklearn as sp
from math import sqrt
import seaborn as sb

# import data from csv
MValdata = pd.read_csv('C:/Users/Kamal/Downloads/Assignment Dataset/student_por.csv', header=0)
# print(MValdata.keys())

#data preperation
feature_cols = ['traveltime','studytime', 'freetime', 'health', 'g2']
x_mul= MValdata[['traveltime','studytime', 'freetime', 'health', 'g2']]
y_mul = np.array(MValdata['g3']).reshape((len(MValdata)),1)
x_mul = np.array(x_mul).reshape((len(x_mul)),5)

x_mul_train, x_mul_test, y_mul_train, y_mul_test = sp.model_selection.train_test_split(x_mul,y_mul, random_state=1)

#initialize and train the model
lr_mul = LinearRegression()
lr_mul.fit(x_mul_train,y_mul_train)

#calculate the linear parameters
# print (lr_mul.coef_)
# print (lr_mul.intercept_)
# #list(zip(feature_cols,lr_mul.coef_))

y_mul_pred = lr_mul.predict(x_mul_test)
print("Root mean squared error ",sqrt(sp.metrics.mean_squared_error(y_mul_test,y_mul_pred)))
print ("R2 Score is", sp.metrics.r2_score(y_mul_test,y_mul_pred))

#cross validateion
scores = sp.model_selection.cross_val_score(lr_mul, y_mul_test,y_mul_pred, cv=5, scoring= 'neg_mean_squared_error')
print ("Root mean squared error for cross validation is ",sqrt(-(scores.mean())))
print ("Standard Deviation is ",scores)

# test=sp.model_selection.cross_validate(lr_mul, y_mul_test,y_mul_pred, cv=5,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
# print (test)


sb.pairplot(MValdata, x_vars=['traveltime','studytime', 'freetime', 'health', 'g2'], y_vars='g3',size=7,aspect=0.7,kind='reg')
pyplot.show()