from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn import utils
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef

class dataset_features_target(object):
    '''
    This class can be used to fee in the dataframe, with features and target values, it can then split
    the x,y, train,test, scale the data using MinMaxScaler (feature range -1, 1). This can then further
    train models with the data, create predictions and then compares the results.
    '''

    def __init__(self, dataframe, features, target):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.dfTrainSplit = []
        self.dfTestSplit = []
        self.scalerFeatures = MinMaxScaler(feature_range=(0, 1))
        self.scalerTarget = MinMaxScaler(feature_range=(0, 1))
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.scalerX = 0
        self.encoderY = preprocessing.LabelEncoder()

    def x_y_train_test_split(self, split_ratio):
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
        encoderY = self.encoderY
        yInversed = encoderY.inverse_transform(y)
        return np.array(yInversed)

    def classification_models(self, clf_names, classifiers):
        xtrain, ytrain, xtest, ytest = self.x_train, self.y_train, self.x_test, self.y_test
        predictions = []
        scores = pd.DataFrame(columns=['name', 'precision (micro)', 'recall (micro)', 'fscore (micro)', 'support (micro)'
                                             , 'precision (macro)', 'recall (macro)', 'fscore (macro)', 'support (macro)'
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


