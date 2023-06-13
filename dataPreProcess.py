import pandas as pd
import pandas_ta as ta
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
pd.set_option("display.max.columns", None)


#read file
dfBTC = pd.read_csv("dataFiles/Binance_BTCUSDT_d.csv", header=1)

# convert the 'date' column to datetime format
dfBTC['Date'] = pd.to_datetime(dfBTC['Date'])
#sort by date
dfSorted = dfBTC.sort_values(by=['Date'], ascending=True)
#reset index
dfBTC = dfSorted.reset_index(drop=True)


# convert the 'date' column to datetime
#create date features
dfBTC['Date'] = pd.to_datetime(dfBTC["Date"])
dfBTC['Year'] = dfBTC['Date'].dt.year
dfBTC['Month'] = dfBTC['Date'].dt.month
dfBTC['Week'] = dfBTC['Date'].dt.isocalendar().week
dfBTC['DayOfWeek'] = dfBTC['Date'].dt.dayofweek
dfBTC['DayOfMonth'] = dfBTC['Date'].dt.day

# prev day close price
dfBTC['PrevClose'] = dfBTC['Close'].shift(1)
#percetange increase/decrease between close price and nextClose price
dfBTC['PercentChange'] = ((dfBTC['Close'] - dfBTC['PrevClose']) / dfBTC['PrevClose']) * 100

#if percentage increase is greater than 0.25% then flag as 2 (bullish)
#if percentage decrease is less than -0.25 then flag as 1 (bearish)
#else 0 (neutral)
#define conditions
conditions = [ dfBTC['PercentChange'] >= 0.25,
               dfBTC['PercentChange'] <= -0.25,
              (dfBTC['PercentChange'] > -0.25) & (dfBTC['PercentChange'] < 0.25) ]
#define results
results = [2, 1, 0]
#create feature
dfBTC['Trend'] = np.select(conditions, results)


#TA indicators
dfBTC['EMA200'] = ta.ema(dfBTC['Close'], 200)
dfBTC['EMA100'] = ta.ema(dfBTC['Close'], 100)
dfBTC['EMA50'] = ta.ema(dfBTC['Close'], 50)
dfBTC['EMA20'] = ta.ema(dfBTC['Close'], 20)
dfBTC['RSI14'] = ta.rsi(dfBTC['Close'], 14)
dfBTC.ta.macd(close='Close', append=True)
dfBTC.ta.adx(close='Close', append=True)


#drop all nan values
dfBTC = dfBTC.dropna()


print(dfBTC)
print(dfBTC['Trend'].value_counts())


#price and ema trend
sns.lineplot(dfBTC, x='Date', y='Close', label='Close')
sns.lineplot(dfBTC, x='Date', y='EMA200', label='EMA200')
sns.lineplot(dfBTC, x='Date', y='EMA100', label='EMA100')
sns.lineplot(dfBTC, x='Date', y='EMA50', label='EMA50')
sns.lineplot(dfBTC, x='Date', y='EMA20', label='EMA20')
plt.legend()
plt.title('Price and EMA trend')
plt.show()


# plot rsi
sns.lineplot(dfBTC, x='Date', y='RSI14')
plt.axhline(y=70, color='r', linestyle='-', label='Over valued')
plt.axhline(y=30, color='g', linestyle='-', label='Under valued')
plt.legend()
plt.show()



#heatmap correlation
# corr_matrix = dfBTC.corr(method='spearman')
# f, ax = plt.subplots(figsize=(16,8))
# mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidth=0.4,
#             annot_kws={"size": 10}, cmap='coolwarm', ax=ax, mask=mask)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title('Feature correlation')
# plt.show()

