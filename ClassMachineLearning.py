from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn import utils
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef

from sklearn.feature_selection import RFE
from operator import itemgetter
class ClassMachineLearning(object):
    '''
    This class can be used to feed in the dataframe, with features and a single target values.
    This can split into x, y train and x,y test.
    Scale the x train and test data using MinMaxScaler (feature range -1, 1).
    label encode y train and test
    This can then further train models with this transformed data, create predictions and then compares the results.
    -----------------------------------------------------------------------------------------------------------------

    Attributes
    ----------
    dataframe : pandas dataframe
         a pandas dataframe
    features : list
         a list of features for input
    target : str
         target feature
    scalerFeatures : MinMaxScaler
         from sklearn.preprocessing import MinMaxScaler
         MinMaxScaler with range from -1 to 1 to scale the features
    x_train : numpy array
         x train split and scaled
    y_train : numpy array
         y train split and use LabelEncoder to convert categorical variables into numerical form
    x_test : numpy array
         x test split and scaled
    y_test : numpy array
         y train split and use LabelEncoder to convert categorical variables into numerical form
    scalerX : scaler of x train
         The scaler used to fit the x train
    encoderY : LabelEncoder
         label encoder used to transform y train and test


    Methods
    -------
    x_y_train_test_split:
        returns the x, y train and test.

    inverse_y:
        returns the original value for the predictions using label encoder

    classification_models:
        returns the score metrics for the various classification models that have been used

    rfe_rank:
        returns a pandas dataframe with ranked features of importance using RFE

    '''

    def __init__(self, dataframe, features: list, target: str):
        """
        Constructor method for the class: ClassMachineLearning
        ...
        :param dataframe : the pandas dataframe to be used
        :type dataframe: pandas dataframe
        :param features : list of all the features to be used as the input
        :type features: list
        :param target : name of the target variable to be predicted
        :type target: str
        """
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.scalerFeatures = MinMaxScaler(feature_range=(-1, 1))
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.scalerX = 0
        self.encoderY = preprocessing.LabelEncoder()

    def x_y_train_test_split(self, split_ratio):
        """
        Splits the data using the specified split ratio going to the train,
        scales the input features and label encodes the target variable.
        ...
        :param split_ratio : the split ratio to go towards the train set e.g. 0.8 == 80%
        :type split_ratio: float
        ...
        :return: returns Xtrain, Ytrain, Xtest, Ytest
        :rtype: numpy arrays
        """
        #split data
        dataframe = self.dataframe
        split = int(len(dataframe) * split_ratio)
        #train split
        train_split = dataframe[:split]
        train_X = train_split[self.features]
        train_Y = train_split[self.target]

        # test split
        test_split = dataframe[split:]
        test_X = test_split[self.features]
        test_Y = test_split[self.target]
        self.dfTestSplit = test_split

        #scalers
        scalerF = self.scalerFeatures
        scalerX = scalerF.fit(train_X)
        self.scalerX = scalerX

        # scale trainX testX
        train_X_scaled = scalerX.transform(train_X)
        test_X_scaled = scalerX.transform(test_X)
        #label encode train and test Y
        encoderY = self.encoderY
        train_Y_encoded = encoderY.fit_transform(train_Y)
        test_Y_encoded = encoderY.fit_transform(test_Y)

        self.x_train, self.y_train = train_X_scaled, train_Y_encoded
        self.x_test, self.y_test = test_X_scaled, test_Y_encoded
        return train_X_scaled, train_Y_encoded, test_X_scaled, test_Y_encoded

    def inverse_y(self, y):
        """
        Inverse the y set i.e. predictions to get the original data
        ...
        :param split_ratio : and array of the predicted values
        :type split_ratio: numpy array
        ...
        :return: returns numpy array of the original values
        :rtype: numpy array
        """
        encoderY = self.encoderY
        yInversed = encoderY.inverse_transform(y)
        return np.array(yInversed)

    def classification_models(self, clf_names, classifiers):
        """
        This loops through all the classification models input via the parameters and compares
        the scores using precision_recall_fscore_support and matthews_corrcoef
        ...
        :param clf_names : text name of the machine models e.g. Nearest Neighbors (k=4)
        :type clf_names: list
        :param classifiers : machine learning model to be used e.g. KNeighborsClassifier(4)
        :type classifiers: list
        ...
        :return: returns a pandas dataframe with the scores and a list of the predicted values
        :rtype: pandas dataframe, list
        """
        xtrain, ytrain, xtest, ytest = self.x_train, self.y_train, self.x_test, self.y_test
        predictions = []
        scores = pd.DataFrame(columns=['Model', 'precision (micro)', 'recall (micro)', 'f1-score (micro)', 'support (micro)'
                                             , 'precision (macro)', 'recall (macro)', 'f1-score (macro)', 'support (macro)'
                                             , 'mcc'])
        for name, clf in zip(clf_names, classifiers):
            #fit model get pred
            print(f'running model: {name}')
            clf.fit(xtrain, ytrain)
            Y_pred = clf.predict(xtest)
            #inverse test and predict y
            Y_pred_inv = self.inverse_y(Y_pred)
            ytest_inv = self.inverse_y(ytest)
            #calc score
            micro = precision_recall_fscore_support(ytest_inv, Y_pred_inv, average="micro")
            macro = precision_recall_fscore_support(ytest_inv, Y_pred_inv, average="macro")
            mcc = matthews_corrcoef(ytest_inv, Y_pred_inv)
            predictions.append([name, Y_pred_inv])
            scores.loc[len(scores)] = [name, micro[0], micro[1], micro[2], micro[3],
                                             macro[0], macro[1], macro[2], macro[3],
                                             mcc]

        return scores, predictions


    def rfe_rank(self, clf_names, classifiers):
        """
        This loops through all the classification models input via the parameters and returns a dataframe
        with ranked features of importance using RFE on the train dataset
        ...
        :param clf_names : text name of the machine models e.g. Decision Tree (Max Depth=7)
        :type clf_names: list
        :param classifiers : machine learning model to be used e.g. DecisionTreeClassifier(max_depth=7, random_state=123)
        :type classifiers: list
        ...
        :return: returns a pandas dataframe with ranked features of importance using RFE
        :rtype: pandas dataframe
        """
        xtrain, ytrain = self.x_train, self.y_train
        featureInputCols = self.features
        dfRanking = pd.DataFrame(columns=['Model', 'Feature', 'Rank'])
        for name, clf in zip(clf_names, classifiers):
            print(f'running model: {name}')

            # Recursive Feature Elimination (RFE)
            n_features_to_select = 1
            rfe = RFE(clf, n_features_to_select=n_features_to_select)
            rfe.fit(xtrain, ytrain)

            # rank the features in order of importance
            for x, y in (sorted(zip(rfe.ranking_, featureInputCols), key=itemgetter(0))):
                rfeList = [name, y, x]
                dfRanking.loc[len(dfRanking)] = rfeList

        return dfRanking

