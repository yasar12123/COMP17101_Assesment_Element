from dataPreProcess import dfBTC
from ClassMachineLearning import ClassMachineLearning
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

#data for input features
df = dfBTC.drop(['Date', 'Unix', 'Symbol', 'NextDayTrend'], axis=1)
featureInputCols = df.columns.to_list()

#specify data into class
dataset = ClassMachineLearning(dfBTC, featureInputCols, ['NextDayTrend'])

#split date into x, y train and test
xtrain, ytrain, xtest, ytest = dataset.x_y_train_test_split(0.8)


#join xtrain with ytrain to df with cols
cols = df.columns.to_list()
cols.append('NextDayTrend')
trainDataWithCols = pd.DataFrame(columns=cols)
for x, y in zip(xtrain, ytrain):
    trainDataWithCols.loc[len(trainDataWithCols)] = np.append(x, y)



# #heatmap correlation
# corr_matrix = trainDataWithCols.corr(method='spearman')
# f, ax = plt.subplots(figsize=(16,8))
# mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidth=0.4,
#             annot_kws={"size": 10}, cmap='coolwarm', ax=ax, mask=mask)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title('Feature correlation')
# plt.show()


# # feature importance using extra trees classifier
# model = ExtraTreesClassifier()
# model.fit(xtrain, ytrain)
# #print with col names
# dfWithCols = pd.DataFrame(columns=featureInputCols)
# dfWithCols.loc[len(dfWithCols)] = model.feature_importances_
# dfWithCols = dfWithCols.reset_index()
# dfColsList = dfWithCols.columns[1:].tolist()
# dfWithCols = pd.melt(dfWithCols, id_vars='index', value_vars=dfColsList)
# dfWithCols = dfWithCols.sort_values(by='value', ascending=False)
# #dfWithCols.to_csv("featureImportanceRank.csv")
# print(dfWithCols)
# #plot scores
# ax = plt.gca()
# sns.barplot(data=dfWithCols, x='value', y='variable')
# for container in ax.containers:
#     ax.bar_label(container)
# plt.title('Feature importance - in correlation with "NextDayTrend"')
# plt.show()






# RFE
#classification models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
max_depths = [5,15,20,40,60,80,100,120]
n_estimator = [100,200,400,800,1000]
classifiers = []
clf_names = []
#insert DecisionTreeClassifiers
for i, x in enumerate(max_depths):
    classifiers.append(DecisionTreeClassifier(max_depth=x, random_state=123))
    clf_names.append(f'DecisionTreeClassifier - max_depth={x}')
#insert RandomForestClassifiers
for i, depth in enumerate(max_depths):
    for a, estimator in enumerate(n_estimator):
        classifiers.append(RandomForestClassifier(n_estimators=estimator, max_depth=depth, random_state=123))
        clf_names.append(f'RandomForestClassifier - n_estimators={estimator}, max_depth={depth}')
classifiers.append(LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=123))
clf_names.append("Logistic Regression")
classifiers.append(xgb.XGBClassifier(objective='multi:softmax', num_class=3))
clf_names.append("xgb")
#print(classifiers)
print(clf_names)

#rfe ranking
rfeRanking = dataset.rfe_rank(clf_names, classifiers)
rfeRanking['Rank'] = -abs(rfeRanking['Rank'])
rfeGroup = rfeRanking.groupby(['Feature'])['Rank'].mean().sort_values()
rfeGroup.plot(kind="barh")
plt.title('Ranking - Mean feature importance')
plt.show()


