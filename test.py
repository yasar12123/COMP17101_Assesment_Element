import pandas as pd
pd.set_option("display.max.columns", None)

dfBTC = pd.read_csv("dataFiles/Binance_BTCUSDT_d.csv", header=1)

# convert the 'date' column to datetime format
dfBTC['Date'] = pd.to_datetime(dfBTC['Date'])
#sort by date
dfSorted = dfBTC.sort_values(by=['Date'], ascending=True)
#reset index
dfBTC = dfSorted.reset_index(drop=True)

print(dfBTC.describe())