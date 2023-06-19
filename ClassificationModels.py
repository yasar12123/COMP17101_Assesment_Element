#from preProcessData import dfDaily
from dataPreProcess import dfBTC
from ClassMachineLearning import ClassMachineLearning

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import math
pd.set_option("display.max.columns", None)

from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


#specify data into class
dataset = ClassMachineLearning(dfBTC, ['Volume BTC', 'PercentChange', 'MACD_12_26_9',
                                       'tradecount', 'ADX_14', 'DMN_14',
                                       'Close', 'MACDh_12_26_9', 'DMP_14'
                                       ], ['NextDayTrend'])

#split date into x, y train and test
xtrain, ytrain, xtest, ytest = dataset.x_y_train_test_split(0.8)

#print unique values in y set
print(f'Unique values in ytrain: {np.unique(ytrain)}')
print(f'Unique values in ytest: {np.unique(ytest)}')
print(f'length of train: {len(xtrain)}')
print(f'length of test: {len(xtest)}')


# #to get best knn
# #method 1 sqrt of n
# k = math.sqrt(len(xtrain))
# print(k)
# #method 2 error rate
# error_rate = []
# for i in range(1,50):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     #knn = DecisionTreeClassifier(max_depth=i, random_state=123)
#     knn.fit(xtrain, ytrain)
#     pred = knn.predict(xtest)
#     error_rate.append(np.mean(pred != ytest))
#
# plt.figure(figsize=(15,10))
# plt.plot(range(1,50),error_rate, marker='o', markersize=9)
# plt.show()


#classificatoin model to use
dummyClass = ['most_frequent', 'stratified', 'uniform']
max_depths = [3, 5, 15]
n_estimator = [100, 300]
knc = [3, 9, 20]
classifiers = []
clf_names = []
#insert DummyClassifier
for i, x in enumerate(dummyClass):
    classifiers.append(DummyClassifier(strategy=x, random_state=123))
    clf_names.append(f'DummyClassifier - {x}')
#insert DecisionTreeClassifiers
for i, x in enumerate(max_depths):
    classifiers.append(DecisionTreeClassifier(max_depth=x, random_state=123))
    clf_names.append(f'DecisionTreeClassifier - max_depth={x}')
#insert RandomForestClassifiers
for i, depth in enumerate(max_depths):
    for ii, estimator in enumerate(n_estimator):
        classifiers.append(RandomForestClassifier(n_estimators=estimator, max_depth=depth, random_state=123))
        clf_names.append(f'RandomForestClassifier - n_estimators={estimator}, max_depth={depth}')
#insert KNeighborsClassifier
for i, x in enumerate(knc):
    classifiers.append(KNeighborsClassifier(x))
    clf_names.append(f'KNeighborsClassifier({x})')

#other models
classifiers.append(LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=123))
clf_names.append("Logistic Regression")
classifiers.append(xgb.XGBClassifier(objective='multi:softmax', num_class=3))
clf_names.append("xgb")
classifiers.append(GaussianNB())
clf_names.append("GaussianNB")
classifiers.append(MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=10000, activation='relu', random_state=123))
clf_names.append("MLP (RelU)")
classifiers.append(LinearSVC(multi_class='ovr', class_weight='balanced', random_state=123))
clf_names.append("LinearSVC")


#get the scores and predictions for the models
scores, predictions = dataset.classification_models(clf_names, classifiers)
print(scores)

#sort scores by f1-score (macro)
scores.sort_values(by=['f1-score (macro)'], ascending=False, inplace=True)

# Create bar plot for scores
ax = plt.gca()
scores.plot(kind='barh', x='Model', y=scores.columns[1:], ax=ax, width=0.9)
for container in ax.containers:
    ax.bar_label(container)
plt.legend()
plt.show()

# Create bar plot for f1-score
ax = plt.gca()
scores.plot(kind='barh', x='Model', y=['f1-score (micro)','f1-score (macro)'], ax=ax, width=0.9)
for container in ax.containers:
    ax.bar_label(container)
plt.legend()
plt.title('f1-score comparison')
plt.show()


#plot confusion matrix
# for x in predictions:
#     cm = confusion_matrix(dataset.inverse_y(ytest),  x[1])
#     cm_df = pd.DataFrame(cm,
#                          index=[2, 1, 0],
#                          columns=['2', '1', '0'])
#
#     # Plotting the confusion matrix
#     plt.figure(figsize=(5, 4))
#     sns.heatmap(cm_df, annot=True)
#     plt.title('Confusion Matrix - ' + x[0])
#     plt.ylabel('Actual Values')
#     plt.xlabel('Predicted Values')
#     plt.show()




