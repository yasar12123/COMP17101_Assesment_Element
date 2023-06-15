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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#specify data into class
dataset = ClassMachineLearning(dfBTC, ['ADX_14', 'tradecount', 'DMP_14',
                                       'Volume BTC', 'MACDh_12_26_9', 'DMN_14',
                                       'PercentChange', 'RSI14', 'EMA200'
                                       #'MACDs_12_26_9', 'MACD_12_26_9', 'Volume USDT',
                                       #'High', 'DayOfMonth'
                                       ], ['NextDayTrend'])

#split date into x, y train and test
xtrain, ytrain, xtest, ytest = dataset.x_y_train_test_split(0.8)

#print unique values in y set
print(f'Unique values in ytrain: {np.unique(ytrain)}')
print(f'Unique values in ytest: {np.unique(ytest)}')
print(f'length of train: {len(xtrain)}')
print(f'length of test: {len(xtest)}')


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


#other models
classifiers.append(KNeighborsClassifier(86))
clf_names.append('KNeighborsClassifier(86)')

classifiers.append(DecisionTreeClassifier(max_depth=3, random_state=123))
clf_names.append('DecisionTreeClassifier - max_depth=3')
classifiers.append(DecisionTreeClassifier(max_depth=4, random_state=123))
clf_names.append('DecisionTreeClassifier - max_depth=4')

classifiers.append(RandomForestClassifier(n_estimators=5, max_depth=300, random_state=123))
clf_names.append('RandomForestClassifier - n_estimators=5, max_depth=300')
classifiers.append(RandomForestClassifier(n_estimators=700, max_depth=10, random_state=123))
clf_names.append('RandomForestClassifier - n_estimators=700, max_depth=10')



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
plt.title('Performance score comparison - optimised')
plt.show()

# Create bar plot for f1-score
ax = plt.gca()
scores.plot(kind='barh', x='Model', y=['f1-score (micro)','f1-score (macro)'], ax=ax, width=0.9)
for container in ax.containers:
    ax.bar_label(container)
plt.legend()
plt.title('f1-score comparison - optimised')
plt.show()


#plot confusion matrix
for x in predictions:
    cm = confusion_matrix(dataset.inverse_y(ytest),  x[1])
    cm_df = pd.DataFrame(cm,
                         index=[2, 1, 0],
                         columns=['2', '1', '0'])

    # Plotting the confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix - ' + x[0])
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()





