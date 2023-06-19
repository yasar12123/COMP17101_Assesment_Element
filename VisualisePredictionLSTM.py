import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

#read file
df = pd.read_csv("LSTM_predictions.csv")

#next day trend
df['TrendForNextDay'] = df['CurrentTrend'].shift(-1)

positive = df[df['Prediction'] == df['TrendForNextDay']]
positiveBuy = positive[positive['Prediction'] == 2]
positiveSell = positive[positive['Prediction'] == 1]
positiveNeutral = positive[positive['Prediction'] == 0]

negatives = df[df['Prediction'] != df['TrendForNextDay']]
negativesBuy = negatives[negatives['Prediction'] == 2]
negativesSell = negatives[negatives['Prediction'] == 1]
negativesNeutral = negatives[negatives['Prediction'] == 0]


#correct with close price
sns.lineplot(df, x='Date', y='Close', color='blue')
plt.plot(positiveBuy['Close'], linestyle='None', marker='^', color='g', label='Buy Signal')
plt.plot(positiveSell['Close'], linestyle='None', marker='v', color='r',  label='Sell Signal')
plt.plot(positiveNeutral['Close'], linestyle='None', marker='.', color='grey',  label='Neutral Signal')
plt.title('BTC/USDT - Close price (test split data) - Correct prediction')
plt.legend()
plt.show()


#incorrect with close price
sns.lineplot(df, x='Date', y='Close', color='blue')
plt.plot(negativesBuy['Close'], linestyle='None', marker='^', color='g', label='Buy Signal')
plt.plot(negativesSell['Close'], linestyle='None', marker='v', color='r',  label='Sell Signal')
plt.plot(negativesNeutral['Close'], linestyle='None', marker='.', color='grey',  label='Neutral Signal')
plt.title('BTC/USDT - Close price (test split data) - Incorrect prediction')
plt.legend()
plt.show()