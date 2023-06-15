import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#read file
testDataWithPrediction = pd.read_csv("LSTM_predictions.csv")


sns.lineplot(testDataWithPrediction, x='Date', y='Close')
plt.title('BTC/USDT - Close price (test split data)')
plt.show()