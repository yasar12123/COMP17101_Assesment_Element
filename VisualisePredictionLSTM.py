import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

#read file
df = pd.read_csv("LSTM_predictions.csv")

conditions = [ df['Prediction'] == 2,
               df['Prediction'] == 1,
               df['Prediction'] == 0 ]
results = ['Bullish', 'Bearish', 'Neutral']
df['PredictionLabel'] = np.select(conditions, results)

positive = df[df['Prediction'] == df['NextDayTrend']]
negatives = df[df['Prediction'] != df['NextDayTrend']]

print(positive)
print(negatives)

colors = ['RED']
sns.lineplot(df, x='Date', y='Close', color='orange')
sns.scatterplot(positive, x='Date', y='Close' , hue='PredictionLabel', palette=colors)
sns.scatterplot(negatives, x='Date', y='Close', hue='PredictionLabel', palette=colors)
plt.title('BTC/USDT - Close price (test split data)')
plt.legend()
plt.show()