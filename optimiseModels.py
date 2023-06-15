#from preProcessData import dfDaily
from dataPreProcess import dfBTC
from ClassMachineLearning import ClassMachineLearning
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import math
pd.set_option("display.max.columns", None)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



#specify data into class
dataset = ClassMachineLearning(dfBTC, ['ADX_14', 'tradecount', 'DMP_14',
                                       'Volume BTC', 'MACDh_12_26_9', 'DMN_14',
                                       'PercentChange', 'RSI14', 'EMA200',
                                       'MACDs_12_26_9', 'MACD_12_26_9', 'Volume USDT',
                                       'High', 'DayOfMonth'
                                       ], ['NextDayTrend'])

#split date into x, y train and test
xtrain, ytrain, xtest, ytest = dataset.x_y_train_test_split(0.8)



#method - error rate for DecisionTreeClassifier
error_rate = []
for i in range(1,200):
    model = KNeighborsClassifier(i)
    model.fit(xtrain, ytrain)
    pred = model.predict(xtest)
    error_rate.append(np.mean(pred != ytest))
plt.figure(figsize=(15,10))
plt.plot(range(1,200),error_rate, marker='o', markersize=9)
plt.title('Error rate - ')
plt.show()

#
# #method - error rate for DecisionTreeClassifier
# error_rate = []
# for i in range(1,100):
#     model = DecisionTreeClassifier(max_depth=i, random_state=123)
#     model.fit(xtrain, ytrain)
#     pred = model.predict(xtest)
#     error_rate.append(np.mean(pred != ytest))
# plt.figure(figsize=(15,10))
# plt.plot(range(1,100),error_rate, marker='o', markersize=9)
# plt.show()
#
# #method - error rate for RandomForestClassifier
# error_rate = []
# for i in range(1,50):
#     model = RandomForestClassifier(max_depth=i)
#     #knn = DecisionTreeClassifier(max_depth=i, random_state=123)
#     model.fit(xtrain, ytrain)
#     pred = model.predict(xtest)
#     error_rate.append(np.mean(pred != ytest))
# plt.figure(figsize=(15,10))
# plt.plot(range(1,50),error_rate, marker='o', markersize=9)
# plt.show()
#
