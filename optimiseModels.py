from dataPreProcess import dfBTC
from ClassMachineLearning import ClassMachineLearning
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
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


#method - error rate for KNC
error_rate = []
for i in range(1,200):
    model = KNeighborsClassifier(i)
    model.fit(xtrain, ytrain)
    pred = model.predict(xtest)
    error_rate.append(np.mean(pred != ytest))
plt.figure(figsize=(15,10))
plt.plot(range(1,200),error_rate, marker='o', markersize=9)
plt.title('Error rate - KNeighborsClassifier ')
plt.show()


#method - error rate for DecisionTreeClassifier
error_rate = []
for i in range(1,100):
    model = DecisionTreeClassifier(max_depth=i, random_state=123)
    model.fit(xtrain, ytrain)
    pred = model.predict(xtest)
    error_rate.append(np.mean(pred != ytest))
plt.figure(figsize=(15,10))
plt.plot(range(1,100),error_rate, marker='o', markersize=9)
plt.title('Error rate - DecisionTreeClassifier ')
plt.show()


#method 1 - error rate for RandomForestClassifier
error_rate = []
depths = [5,10,30,50,80,100,150,200,250,300,350,400, 500,700,1000]
estimators = [5,10,30,50,80,100,150,200,250,300,350,400, 500,700,1000]
xLabel = []
for i, depth in enumerate(depths):
    for ii, estimator in enumerate(estimators):
        model = RandomForestClassifier(n_estimators=estimator, max_depth=depth, random_state=123)
        model.fit(xtrain, ytrain)
        pred = model.predict(xtest)
        error_rate.append(np.mean(pred != ytest))
        xLabel.append(str((depth,estimator)))
plt.figure(figsize=(15,10))
plt.plot(xLabel, error_rate, marker='o', markersize=9)
plt.title('Error rate - RandomForestClassifier ')
plt.show()


# #method 2 - error rate for RandomForestClassifier - n_estimators
# #n_est, maxDep = 700, 10
# rangeList = [5,10,30,50,80,100,150,200,250,300,350,400, 500,700,1000]
# error_rate = []
# for i in rangeList:
#     model = RandomForestClassifier(n_estimators=700, max_depth=i ,random_state=123)
#     model.fit(xtrain, ytrain)
#     pred = model.predict(xtest)
#     error_rate.append(np.mean(pred != ytest))
# plt.figure(figsize=(15,10))
# plt.plot(rangeList, error_rate, marker='o', markersize=9)
#
# plt.title('Error rate - RandomForestClassifier(n_estimators=700) - max_depth ')
# plt.show()
