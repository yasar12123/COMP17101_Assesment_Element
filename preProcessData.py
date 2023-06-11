import numpy as np
import pandas as pd
import pandas_ta as ta
import seaborn as sns
from matplotlib import pyplot as plt

#Read the csv file
dfRaw = pd.read_csv("Gemini_BTCUSD_1h.csv", header=1)

# convert the 'date' column to datetime format
dfRaw['datetime'] = pd.to_datetime(dfRaw["date"])

#sort by date
dfSorted = dfRaw.sort_values(by=["datetime"], ascending=True)
#reset index
dfSorted = dfSorted.reset_index(drop=True)
#set datetime as index
#dfSorted = dfSorted.set_index('datetime')


#remove columns
dfSorted.drop('unix', axis=1, inplace=True)
dfSorted.drop('symbol', axis=1, inplace=True)
dfSorted.drop('date', axis=1, inplace=True)

#create date/time features
dfSorted['Date'] = dfSorted['datetime'].dt.date
dfSorted['Hour'] = dfSorted['datetime'].dt.hour

#Daily table
#aggregate into daily table
aggregations = {'open': 'first',
                'close': 'last',
                'high': 'max',
                'low': 'min',
                'Volume BTC': 'sum',
                'Volume USD': 'sum'}
dfDaily = dfSorted.groupby('Date', as_index=False).agg(aggregations)

#create date features
# convert the 'date' column to datetime format
dfDaily['Date'] = pd.to_datetime(dfDaily["Date"])
dfDaily['Year'] = dfDaily['Date'].dt.year
dfDaily['Month'] = dfDaily['Date'].dt.month
dfDaily['Week'] = dfDaily['Date'].dt.isocalendar().week
dfDaily['DayOfWeek'] = dfDaily['Date'].dt.dayofweek
dfDaily['DayOfMonth'] = dfDaily['Date'].dt.day


#next day close price
dfDaily['NextClose'] = dfDaily['close'].shift(-2)
#percetange increase/decrease between close price and nextClose price
dfDaily['NextPercentChange'] = ((dfDaily['NextClose'] - dfDaily['close']) / dfDaily['close']) * 100

#if percentage increase is greater than 0.25% then flag as 3 (bullish)
#if percentage decrease is less than -0.25 then flag as 2 (bearish)
#else 1 (neutral)
#define conditions
conditions = [ dfDaily['NextPercentChange'] >= 0.25,
               dfDaily['NextPercentChange'] <= -0.25,
              (dfDaily['NextPercentChange'] > -0.25) & (dfDaily['NextPercentChange'] < 0.25) ]
#define results
results = [2, 1, 0]
#create feature
dfDaily['BullishBearish'] = np.select(conditions, results)



#TA indicators
dfDaily['RSI14'] = ta.rsi(dfDaily['close'], 14)
dfDaily['EMA200'] = ta.ema(dfDaily['close'], 200)
dfDaily['EMA100'] = ta.ema(dfDaily['close'], 100)
dfDaily['EMA50'] = ta.ema(dfDaily['close'], 50)
dfDaily['EMA20'] = ta.ema(dfDaily['close'], 20)
dfDaily['EMA10'] = ta.ema(dfDaily['close'], 10)
dfDaily.ta.stoch(high='high', low='low', k=14, d=3, append=True)
dfDaily.ta.macd(close='close', append=True)
dfDaily.ta.adx(close='close', append=True)


#drop all nan values
dfDaily = dfDaily.dropna()

#view all columns
pd.set_option("display.max.columns", None)
print(dfDaily)
print(dfDaily['BullishBearish'].value_counts())



#plots
# sns.boxplot( x=dfDaily['BullishBearish'], y=dfDaily['RSI14'] )
# plt.show()
#
# #heatmap correlation
# corr_matrix = dfDaily.corr(method='spearman')
# f, ax = plt.subplots(figsize=(16,8))
# sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidth=0.4,
#             annot_kws={"size": 10}, cmap='coolwarm', ax=ax)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.show()

