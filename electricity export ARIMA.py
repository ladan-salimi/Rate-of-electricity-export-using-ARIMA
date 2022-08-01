#Rate of electricity export using ARIMA
from dateutil.parser import parse 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from numpy import log

# Import as Dataframe
df = pd.read_excel(r'C:\Users\ladan\Desktop\clustring\time series pattern recognition.xlsx',sheet_name=2,parse_dates=['date'], index_col='date')
# Data Plot
def plot_df(df, x, y, title="Data Plot", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
plot_df(df, x=df.index, y=df.value, title='yearly export electricity subbmitted by NO to DK.')
############# autocorrelation plot
autocorrelation_plot(df)
plt.show()
#######stationary test
results = adfuller(df)
###########non stationary convert to stationary
dataframe = DataFrame(df.values)
dataframe.columns = ['value']
dataframe['value'] = log(dataframe['value'])
####################seasonal_decompose
decompose = seasonal_decompose(df['value'],model='additive')
decompose.plot()
plt.show()
#################seperation of training and test data to apply arrima
df['date'] = df.index
train = df[df['date'] < pd.to_datetime("2021-09")]
train['train'] = train['value']
del train['date']
del train['value']
test = df[df['date'] >= pd.to_datetime("2021-09")]
del test['date']
test['test'] = test['value']
del test['value']
plt.plot(train, color = "black")
plt.plot(test, color = "blue")
plt.title("Train/Test split for export Data")
plt.ylabel("export rate")
plt.xlabel('Year-Month')
sns.set()
plt.show()
###############prediction
model = ARIMA(train,order=[1,0,1])
model=model.fit()
model.summary()
train_size=len(train)
test_size=len(test)
end=train_size+test_size-1
pred=model.predict(start=train_size, end=end)
print(pred)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(pred, color='red')
pyplot.show()
#############rmse
rms = sqrt(mean_squared_error(test,pred))
print("RMSE: ", rms)
############Reversal of prediction results (from log data to actual ones)
arr=pred.to_numpy()
rev1=np.exp(arr)
print(rev1)
arr=test.to_numpy()
rev2=np.exp(arr)
print(rev2)
############# plot forecasts against actual outcomes
pyplot.plot(rev1)
pyplot.plot(rev2, color='red')
pyplot.show()
##########prediction for next 5 month data
com_pred=model.predict(start=start, end=end+5)
print(com_pred)
