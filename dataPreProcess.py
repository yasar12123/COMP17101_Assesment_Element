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

# CurrentTrend
#if percentage increase is greater than 0.25% then flag as 2 (bullish)
#if percentage decrease is less than -0.25 then flag as 1 (bearish)
#else 0 (neutral)
#define conditions
conditions = [ dfBTC['PercentChange'] >= 0.25,
               dfBTC['PercentChange'] <= -0.25,
              (dfBTC['PercentChange'] > -0.25) & (dfBTC['PercentChange'] < 0.25) ]
results = [2, 1, 0]
dfBTC['CurrentTrend'] = np.select(conditions, results)


#Log transform
dfBTC['Open'] = np.log(dfBTC['Open'])
dfBTC['High'] = np.log(dfBTC['High'])
dfBTC['Low'] = np.log(dfBTC['Low'])
dfBTC['Close'] = np.log(dfBTC['Close'])
dfBTC['Volume BTC'] = np.log(dfBTC['Volume BTC'])
dfBTC['Volume USDT'] = np.log(dfBTC['Volume USDT'])
dfBTC['PrevClose'] = np.log(dfBTC['PrevClose'])
#dfBTC['PercentChange'] = np.log(dfBTC['PercentChange'])
dfBTC['tradecount'] = np.log(dfBTC['tradecount'])


#TA indicators
dfBTC['EMA200'] = ta.ema(dfBTC['Close'], 200)
dfBTC['EMA100'] = ta.ema(dfBTC['Close'], 100)
dfBTC['EMA50'] = ta.ema(dfBTC['Close'], 50)
dfBTC['EMA20'] = ta.ema(dfBTC['Close'], 20)
dfBTC['RSI14'] = ta.rsi(dfBTC['Close'], 14)
dfBTC.ta.macd(close='Close', append=True)
dfBTC.ta.adx(close='Close', append=True)

#next day trend
dfBTC['NextDayTrend'] = dfBTC['CurrentTrend'].shift(-1)

#drop all nan values
dfBTC = dfBTC.dropna()

# #view data
# print(dfBTC)
# print(dfBTC['CurrentTrend'].value_counts())
#print(dfBTC.columns)
#print(len(dfBTC))

#
# #price and ema trend
# sns.lineplot(dfBTC, x='Date', y='Close', label='Close')
# sns.lineplot(dfBTC, x='Date', y='EMA200', label='EMA200')
# sns.lineplot(dfBTC, x='Date', y='EMA100', label='EMA100')
# sns.lineplot(dfBTC, x='Date', y='EMA50', label='EMA50')
# sns.lineplot(dfBTC, x='Date', y='EMA20', label='EMA20')
# plt.legend()
# plt.title('Price and EMA trend')
# plt.show()
#
#
# # plot rsi
# sns.lineplot(dfBTC, x='Date', y='RSI14')
# plt.axhline(y=70, color='r', linestyle='-', label='Over valued - 70')
# plt.axhline(y=30, color='g', linestyle='-', label='Under valued - 30')
# plt.title('RSI 14')
# plt.legend()
# plt.show()



# #sub plots of distribution
# fig, axes = plt.subplots(5, 4)
# sns.distplot(ax=axes[0, 0], x=dfBTC['Open'])
# axes[0, 0].set_ylabel('Open')
# sns.distplot(ax=axes[0, 1], x=dfBTC['High'])
# axes[0, 1].set_ylabel('High')
# sns.distplot(ax=axes[0, 2], x=dfBTC['Low'])
# axes[0, 2].set_ylabel('Low')
# sns.distplot(ax=axes[0, 3], x=dfBTC['Close'])
# axes[0, 3].set_ylabel('Close')
# sns.distplot(ax=axes[1, 0], x=dfBTC['Volume BTC'])
# axes[1, 0].set_ylabel('Volume BTC')
# sns.distplot(ax=axes[1, 1], x=dfBTC['Volume USDT'])
# axes[1, 1].set_ylabel('Volume USDT')
# sns.distplot(ax=axes[1, 2], x=dfBTC['tradecount'])
# axes[1, 2].set_ylabel('tradecount')
# sns.distplot(ax=axes[1, 3], x=dfBTC['PrevClose'])
# axes[1, 3].set_ylabel('PrevClose')
# sns.distplot(ax=axes[2, 0], x=dfBTC['PercentChange'])
# axes[2, 0].set_ylabel('PercentChange')
# sns.distplot(ax=axes[2, 1], x=dfBTC['EMA200'])
# axes[2, 1].set_ylabel('EMA200')
# sns.distplot(ax=axes[2, 2], x=dfBTC['EMA100'])
# axes[2, 2].set_ylabel('EMA100')
# sns.distplot(ax=axes[2, 3], x=dfBTC['EMA50'])
# axes[2, 3].set_ylabel('EMA50')
# sns.distplot(ax=axes[3, 0], x=dfBTC['EMA20'])
# axes[3, 0].set_ylabel('EMA20')
# sns.distplot(ax=axes[3, 1], x=dfBTC['RSI14'])
# axes[3, 1].set_ylabel('RSI14')
# axes[3, 1].set_yticklabels([])
# sns.distplot(ax=axes[3, 2], x=dfBTC['MACD_12_26_9'])
# axes[3, 2].set_ylabel('MACD_12_26_9')
# axes[3, 2].set_yticklabels([])
# sns.distplot(ax=axes[3, 3], x=dfBTC['MACDh_12_26_9'])
# axes[3, 3].set_ylabel('MACDh_12_26_9')
# axes[3, 3].set_yticklabels([])
# sns.distplot(ax=axes[4, 0], x=dfBTC['MACDs_12_26_9'])
# axes[4, 0].set_ylabel('MACDs_12_26_9')
# axes[4, 0].set_yticklabels([])
# sns.distplot(ax=axes[4, 1], x=dfBTC['ADX_14'])
# axes[4, 1].set_ylabel('ADX_14')
# axes[4, 1].set_yticklabels([])
# sns.distplot(ax=axes[4, 2], x=dfBTC['DMP_14'])
# axes[4, 2].set_ylabel('DMP_14')
# axes[4, 2].set_yticklabels([])
# sns.distplot(ax=axes[4, 3], x=dfBTC['DMN_14'])
# axes[4, 3].set_ylabel('DMN_14')
# axes[4, 3].set_yticklabels([])
# fig.suptitle('Data distribution post log transform')
# plt.show()


