import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import datasets, linear_model
from sklearn.svm import SVR
import sklearn as sp
import seaborn as sb
from math import sqrt

#import the data from csv file
SVCData = pd.read_csv('C:/Users/Kamal/Downloads/Assignment Dataset/student_por.csv', header=0)

#initialize the model object
svr_lin = SVR(kernel='linear')

#data preperation
x_lin= np.array(SVCData.g2).reshape((len(SVCData.g2), 1))
y_lin = np.array(SVCData.g3)

x_train, x_test, y_train, y_test = sp.model_selection.train_test_split(x_lin,y_lin, random_state=1)

#train the model with training data
svr_lin.fit(x_train,y_train)


#calculate Root mean squared error
y_pred = svr_lin.predict(x_test)
print("Root mean squared error ",sqrt(sp.metrics.mean_squared_error(y_test,y_pred)))

#cross validation with 5 folds
scores = sp.model_selection.cross_val_score(svr_lin, x_train,y_train, cv=5, scoring= 'neg_mean_squared_error')
print ("Root mean squared error for crossValidation is ",sqrt(-(scores.mean())))


#poly model for experimentation
svr_poly = SVR(kernel='poly')
svr_poly.fit(x_train,y_train)
print("Root mean squared error for polynomial is ",sqrt(sp.metrics.mean_squared_error(y_test,svr_poly.predict(x_test))))
scoresPoly = sp.model_selection.cross_val_score(svr_poly, x_train,svr_poly.predict(x_train), cv=5, scoring= 'neg_mean_squared_error')
# print ("Root mean squared error for crossValidation for polynomial  is ",sqrt(-(scoresPoly.mean())))

#plot the predictaed values
pyplot.scatter(x_lin,y_lin,color='black',label='data')
pyplot.plot(x_lin,svr_lin.predict(x_lin),color='blue',label='Linear SVR')
pyplot.plot(x_lin,svr_poly.predict(x_lin),color='red',label='Poly SVR')
pyplot.xlabel('g2')
pyplot.ylabel('g3')
pyplot.title('Support Vector Regression')
pyplot.legend()
pyplot.show()