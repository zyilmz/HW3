import pandas as pd 
import numpy as np
import os 
import time
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

os.listdir()
data=pd.read_csv("usdtrytrading.txt",sep="\t")
print(data.shape)
data=data.dropna()
# data=data[data.Volume>0]
print(data.shape)

data.head()
print(data.shape)
data.head()
data_May_6=data.iloc[-942:,:]
print(data_May_6.head())
df_orig=data['Close']
df_may6=data_May_6['Close']

def plot_data(data,running_average_window,tit):
    
    fig= plt.figure()
    rolling_mean = data.rolling(window=running_average_window).mean()
    rolling_mean2 = data.rolling(window=running_average_window).std()
    
    print(rolling_mean.shape)
    plt.plot(data, label=tit)
    plt.plot(rolling_mean, label=str(running_average_window)+' Sample Average of Mean', color='orange')
#     plt.plot(rolling_mean2, label=str(running_average_window)+' Minutes Average of Std', color='orange')
#     plt.plot(data, rolling_mean2, label='AMD 50 Day SMA', color='magenta')
    plt.legend(loc='upper left')
    plt.title(tit+" and "+str(running_average_window)+" Sample average")
    plt.show()
    
    fig.savefig(tit+"_"+str(running_average_window)+".png")
    fig2=plt.figure()
    plt.plot(rolling_mean2, label=str(running_average_window)+' Sample Average of Std', color='orange')
    plt.title("Std of "+tit)
    plt.show()
    
	
plot_data(df_orig,20,'Original Data')
plot_data(df_orig,50,'Original Data')
plot_data(df_may6,20,'May 6')
plot_data(df_may6,50,'May 6')

def dataLoader_OLS(data):
    train_features=data.iloc[:-2,2:]
    train_labels=data.iloc[2:,5]
    test_features=data.iloc[-2:,2:]
    return train_features,train_labels,test_features
	
X,y,X_test=dataLoader_OLS(data_May_6[data_May_6.Volume>0])
# X,y,X_test=dataLoader_OLS(data_May_6)

regr = linear_model.LinearRegression()

# Train the model using the training sets
# regr.fit(X.iloc[-10:,:], y.iloc[-10:])
regr.fit(X.fillna(0),y)
# Make predictions using the testing set
y_pred = regr.predict(X.fillna(0))
MSE=sklearn.metrics.mean_squared_error(y, y_pred)
print("MSE for OLS is ",str(MSE))

y_pred=regr.predict(X_test)
print("Estimations for 15.59 and 16.00 are",y_pred)


reg = linear_model.Ridge(alpha=.5)
reg.fit(X.fillna(0), y)

# Make predictions using the testing set
y_pred = reg.predict(X.fillna(0))
MSE=sklearn.metrics.mean_squared_error(y, y_pred)
print("MSE for Ridge is ",str(MSE))

y_pred=reg.predict(X_test)
print("Estimations for 15.59 and 16.00 are",y_pred)

alldata=pd.read_csv("usdtrytrading.txt",sep="\t")
data=alldata.iloc[::60,:]
print(data)
# data=data[data.Volume>0]
# print(data.iloc[:100,:])

X,y,X_test2=dataLoader_OLS(data)
print(X.shape)
print(X)

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X, y)

# Make predictions using the testing set
y_pred = regr.predict(X)
MSE=sklearn.metrics.mean_squared_error(y, y_pred)
print("MSE for OLS is ",str(MSE))

y_pred=regr.predict(X_test)
print("Estimations for 15.59 and 16.00 are",y_pred)

reg = linear_model.Ridge(alpha=.5)
reg.fit(X.fillna(0), y)

# Make predictions using the testing set
y_pred = reg.predict(X.fillna(0))
MSE=sklearn.metrics.mean_squared_error(y, y_pred)
print("MSE for Ridge is ",str(MSE))

y_pred=reg.predict(X_test)
print("Estimations for 15.59 and 16.00 are",y_pred)

plot_data(data.iloc[:,3],20,'Hourly Data')
plot_data(data.iloc[:,3],50,'Hourly Data')
