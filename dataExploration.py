import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
pd.set_option("display.max.columns", None)

#read file
dfBTC = pd.read_csv("dataFiles/Binance_BTCUSDT_d.csv", header=1)

#data types
print(dfBTC.dtypes)

# convert the 'date' column to datetime format
dfBTC['Date'] = pd.to_datetime(dfBTC['Date'])
#sort by date
dfSorted = dfBTC.sort_values(by=['Date'], ascending=True)
#reset index
dfBTC = dfSorted.reset_index(drop=True)

print(dfBTC.describe())
#dfBTC.describe().to_csv("dataDescription.csv")

#distribution of price features
sns.kdeplot(dfBTC['Open'])
sns.kdeplot(dfBTC['High'])
sns.kdeplot(dfBTC['Low'])
sns.kdeplot(dfBTC['Close'])
plt.legend(labels=['Open', 'High', 'Low', 'Close'])
plt.xlabel('OHLC')
plt.title('OHLC Distribution')
plt.show()


#pairplot - distribution between features
sns.pairplot(dfBTC, diag_kind = 'kde')
plt.show()


#heatmap correlation
corr_matrix = dfBTC.corr(method='spearman')
f, ax = plt.subplots(figsize=(16,8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidth=0.4,
            annot_kws={"size": 10}, cmap='coolwarm', ax=ax, mask=mask)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title('Feature correlation')
plt.show()


