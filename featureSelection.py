from dataPreProcess import dfBTC
from ClassMachineLearning import ClassMachineLearning

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns



featureInputCols = dfBTC.columns[3:-1].to_list()
#specify data into class
dataset = ClassMachineLearning(dfBTC, featureInputCols, ['NextDayTrend'])

#split date into x, y train and test
xtrain, ytrain, xtest, ytest = dataset.x_y_train_test_split(0.8)


# feature extraction using extra trees classifier
model = ExtraTreesClassifier(n_estimators=50)
model.fit(xtrain, ytrain)
#print with col names
dfWithCols = pd.DataFrame(columns=featureInputCols)
dfWithCols.loc[len(dfWithCols)] = model.feature_importances_
print(dfWithCols)



# RFE using random fores classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from operator import itemgetter

rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=123)

# Recursive Feature Elimination (RFE)
n_features_to_select = 1
rfe = RFE(rfc, n_features_to_select=n_features_to_select)
rfe.fit(xtrain, ytrain)

#rank the features in order of importance
for x, y in (sorted(zip(rfe.ranking_ , featureInputCols), key=itemgetter(0))):
    print(x, y)