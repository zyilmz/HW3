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
print(data.iloc[:,6])

def replace_with_LWMA(data):
    newData=data.copy()
    for i in range(data.shape[0]-60):
#     for i in range(100):
        for j in range(5):
            window=data.iloc[i:i+60,j+2]
            weight=data.iloc[i:i+60,6]
            weight=weight/np.sum(weight)
            newData.iloc[i+60,j+2]=np.sum(weight*window)
            
#         newData['Open']=data['Open'].rolling(window=60).mean()
#         newData['Close']=data['Close'].rolling(window=60).mean()
#         newData['High']=data['High'].rolling(window=60).mean()
#         newData['Low']=data['Low'].rolling(window=60).mean()
#         newData['Volume']=data['Volume'].rolling(window=60).mean()
#         newData.dropna(inplace=True)
    newData.shape
    return newData
	
	
def replace_with_MA(data):
    newData=data.copy()
    newData['Open']=data['Open'].rolling(window=60).mean()
    newData['Close']=data['Close'].rolling(window=60).mean()
    newData['High']=data['High'].rolling(window=60).mean()
    newData['Low']=data['Low'].rolling(window=60).mean()
    newData['Volume']=data['Volume'].rolling(window=60).mean()
    newData.dropna(inplace=True)
    newData.shape
    return newData
	
	
def replace_with_random_sample(data):
    newData=data.copy()
    start_ind=0;
    while start_ind<data.shape[0]-60 :
        a=start_ind+np.random.randint(60)
        for j in range(60):
            newData.iloc[start_ind+j]=newData.iloc[a]
        start_ind=start_ind+60
    newData.dropna(inplace=True)
    newData.shape
    return newData
	
MAData=replace_with_MA(data)
LWMAData=replace_with_LWMA(data)
RSData=replace_with_random_sample(data)

def plot_data(data,running_average_window,tit):
    
    fig= plt.figure()
    rolling_mean = data.rolling(window=running_average_window).mean()
    rolling_mean2 = data.rolling(window=running_average_window).std()
    
#     print(rolling_mean.shape)
    plt.plot(data, label=tit)
#     plt.plot(rolling_mean, label=str(running_average_window)+' Sample Average of Mean', color='orange')
#     plt.plot(rolling_mean2, label=str(running_average_window)+' Minutes Average of Std', color='orange')
#     plt.plot(data, rolling_mean2, label='AMD 50 Day SMA', color='magenta')
    plt.legend(loc='upper left')
    plt.title(tit)
    plt.show()
    
    fig.savefig(tit+".png")
	
plot_data(RSData['Close'],60,'random sample')
plot_data(MAData['Close'],60,'moving average')
plot_data(LWMAData['Close'],60,' linearly weighted moving average')


import sklearn
def regressionResult(data):
    train_features=data.iloc[:-2,2:]
    train_labels=data.iloc[2:,5]
    test_features=data.iloc[-2:,2:]
    
    regr = linear_model.LinearRegression()
    regr.fit(train_features.fillna(0),train_labels)
    y_pred = regr.predict(train_features.fillna(0))
    MSE=sklearn.metrics.mean_squared_error(train_labels, y_pred)
    print("MSE for OLS is ",str(MSE))

    y_pred=regr.predict(test_features)
    print("Estimations for 15.59 and 16.00 are",y_pred)
    
    reg = linear_model.Ridge(alpha=.5)
    reg.fit(train_features.fillna(0), train_labels)

    # Make predictions using the testing set
    y_pred = reg.predict(train_features.fillna(0))
    MSE=sklearn.metrics.mean_squared_error(train_labels, y_pred)
    print("MSE for Ridge is ",str(MSE))

    y_pred=reg.predict(test_features)
    print("Estimations for 15.59 and 16.00 are",y_pred)
    #     return train_features,train_labels,test_features
	
print("MAData")
regressionResult(MAData)
print("--------")

print("LWMAData")
regressionResult(LWMAData)
print("--------")

print("RSData")
regressionResult(RSData)
print("--------")