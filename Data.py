import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pandas_ta as ta

#Read the csv file
dfBTC = pd.read_csv("dataFiles/Binance_BTCUSDT_d.csv", header=1)
dfETH = pd.read_csv("dataFiles/Binance_ETHUSDT_d.csv", header=1)
dfXRP = pd.read_csv("dataFiles/Binance_XRPUSDT_d.csv", header=1)

#cov to date and order by ascending
def dateOrder(df, col_name):
    # convert the 'date' column to datetime format
    df[col_name] = pd.to_datetime(df[col_name])
    #sort by date
    dfSorted = df.sort_values(by=[col_name], ascending=True)
    #reset index
    dfSorted = dfSorted.reset_index(drop=True)
    return dfSorted

dfBTC = dateOrder(dfBTC, 'Date')
dfETH = dateOrder(dfETH, 'Date')
dfXRP = dateOrder(dfXRP, 'Date')

#add technical indicators
def addTi(df):
    # TA indicators
    df['RSI14'] = ta.rsi(df['Close'], 14)
    df['EMA200'] = ta.ema(df['Close'], 200)
    df['EMA100'] = ta.ema(df['Close'], 100)
    df['EMA50'] = ta.ema(df['Close'], 50)
    df['EMA20'] = ta.ema(df['Close'], 20)
    df['EMA10'] = ta.ema(df['Close'], 10)
    df.ta.stoch(high='High', low='Low', k=14, d=3, append=True)
    df.ta.macd(close='Close', append=True)
    df.ta.adx(close='Close', append=True)
    return df

dfBTC = addTi(dfBTC)
dfETH = addTi(dfETH)
dfXRP = addTi(dfXRP)


#Rename and drop cols
def renameColumns(df, name, dropCols):
    df.drop(dropCols, axis=1, inplace=True)
    for columns in df.columns:
        df.rename(columns={columns: name+columns}, inplace=True)
        df.rename(columns={name+'Date': 'Date'}, inplace=True)
    return df

dfBTC = renameColumns(dfBTC, 'BTC_', ['Unix','Symbol'])
dfETH = renameColumns(dfETH, 'ETH_', ['Unix','Symbol','Open','High','Low','Close'])
dfXRP = renameColumns(dfXRP, 'XRP_', ['Unix','Symbol','Open','High','Low','Close'])


#join dfs
def mergeDf(dfs, joinCol):
    finalDf = pd.DataFrame(columns=['Date'])
    for index, df in enumerate(dfs):
       finalDf = finalDf.merge(df, on=joinCol, how='right')
    return finalDf

dfCrypto = mergeDf([dfBTC, dfETH, dfXRP], 'Date')

#drop nan values
dfCrypto.dropna(inplace=True)



#print df
pd.set_option("display.max.columns", None)
print(len(dfCrypto))
print(dfCrypto.dtypes)
#print(dfCrypto.describe())

# #heatmap correlation
# corr_matrix = dfCrypto.corr(method='spearman')
# f, ax = plt.subplots(figsize=(30,30))
# sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidth=0.4,
#             annot_kws={"size": 10}, cmap='coolwarm', ax=ax)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.show()


