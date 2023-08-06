#!/usr/bin/env python
# coding: utf-8

# ## The following code does an analysis on stock price of Microsoft and makes a buying and seling decision based on 15ma and 200 #ma rule
# ### 1. Stock price of 5 years from 2015 to 2020 is fetched via Alpha vantage API
# ### 2. The buying decision is taken when the 15 ma signal crosses 200 ma signal from above
# ### 3. The selling action is taken when 15 ma signal crosses the 200 ma signal from below
# ### 4. Finding the final wealth of the trader, for both strategies, to an accuracy of two (2) decimal points
# ### 5. Plotting the outcome of the trader

# In[32]:


from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt


# In[33]:


api_key = '' # provide user api key
stock = 'MS'
start_dt = np.datetime64('2015-01-01')
end_dt = np.datetime64('2020-12-31')
long_MA = 200 #days
short_MA = 15 #days
initial_wealth = 1000


# ## Make use of the alpha vantage module to download the prices of stock from start_dt to end_dt.

# In[34]:


def slice_series(dt_frm, start_date, end_date):
#Return sliced time series based on the start ate and end date provided
    data_slice1 = dt_frm[dt_frm.index > start_date] 
    data_slice2 = data_slice1[data_slice1.index <end_dt] 
    return data_slice2 


# In[46]:


ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily('MS', outputsize='full')
dt_data = slice_series(data,start_dt,end_dt)
print(f'The stock data of {stock} from {start_dt} to {end_dt} is as follows.')
dt_data


# ## Assumptions made:
# 
# ### 1. The trading strategy is assumed to be without **human intervention**
# 
# ### 2. The buying decision is taken when the 15 ma signal crosses 200 ma signal from above(sensing that the stock is undervalued now). If the price continues to fall, no discretion is made and the stock is hold. 
# 
# ### 3. The selling action is taken when 15 ma signal crosses the 200 ma signal from below(sensing that the stock is overvalued now)
# 
# ### 4. The buy and sell will be done automatically even if the transaction ends in loss. No other strategies is considered
# 

# In[36]:


ti = TechIndicators(key=api_key, output_format='pandas')
#downloading 15 ma 
data_ti, meta_data_ti = ti.get_sma(symbol='MS', interval='daily',
                                    time_period=short_MA, series_type='close')
#downloading 200 ma 
data_t2, meta_data_t2 = ti.get_sma(symbol='MS', interval='daily',                                   
                                    time_period=long_MA, series_type='close')


# In[49]:


#slicing the data with the desired time period
dt_short_sma = slice_series(data_ti,start_dt,end_dt)
dt_long_sma = slice_series(data_t2,start_dt,end_dt)

#renaming the colums to plot
df1 = dt_short_sma
df1 = df1.rename(columns={'SMA': 'short ma'})
df2 = dt_long_sma
df2 = df2.rename(columns={'SMA': 'long ma'})

# ploting the trading signals
total_df = pd.concat([df2,df1], axis=1)
fig = plt.figure(figsize= [15,4])
total_df.plot()
plt.title('Trading Signal')
plt.show()


# In[44]:


#Adding closing price to the moving averages
total_df = pd.concat([dt_data['4. close'], df1,df2], axis=1)
total_df = total_df.rename(columns={'4. close': 'Closing price'})

#Adding signal to the series
len_total = len(total_df.index)
df_signal = total_df.assign(signal = np.zeros(len_total))
df_signal.loc[df_signal['short ma'] < df_signal['long ma'] , 'signal'] = 1

#creating buy signal
df_signal = df_signal.assign(buysignal = np.zeros(len_total))
df_signal = df_signal.assign(sellsignal = np.zeros(len_total))
for i in range(1,len_total) :
    df_signal.iloc[i,4] = df_signal.iloc[i, 3]-df_signal.iloc[i-1, 3]
df_signal.loc[df_signal['buysignal'] == -1, 'buysignal'] = 0

#creating sell signal
for i in range(len_total) :
    df_signal.iloc[i,5] = df_signal.iloc[i, 3]-df_signal.iloc[i-1, 3]
df_signal.loc[df_signal['sellsignal'] == 1, 'sellsignal'] = 0
df_signal.loc[df_signal['sellsignal'] == -1, 'sellsignal'] = 1

#finding buying and selling dates
sell_dates = df_signal[df_signal['sellsignal']==1].index
buy_dates = df_signal[df_signal['buysignal']==1].index
buy_sell_detail = pd.DataFrame()
buy_sell_detail = buy_sell_detail.assign(Buy_Date = buy_dates)
buy_sell_detail = buy_sell_detail.assign(Buy_Price = df_signal.loc[buy_dates]['Closing price'].tolist())
buy_sell_detail = buy_sell_detail.assign(Sell_Date = sell_dates)
buy_sell_detail = buy_sell_detail.assign(Sell_Price = df_signal.loc[sell_dates]['Closing price'].tolist())
print('The buy and sell dates and corresponding prices will be as shown below')
print(buy_sell_detail)


# 
# ## Finding the final wealth of the trader, for both strategies, to an accuracy of two (2) decimal points

# In[39]:


#Creating BUY LOW and SELL HIGH dates
#initial wealth is $1000 till the first buy and sell happens.
df_signal = df_signal.assign(wealth_ma = np.zeros(len_total))
wealth_ma = initial_wealth
num_buy_sell = len(buy_sell_detail)

#intialising intial_wealth as accumulated wealth till first signal is observed
first_buy = buy_sell_detail['Buy_Date'][0]
df_signal.loc[df_signal.index < first_buy , 'wealth_ma'] = wealth_ma

#Calculating the wealth 
for i in range(num_buy_sell):
    if(i>0):
        last_sell = buy_sell_detail['Sell_Date'][i-1]
        df_signal.loc[df_signal.index > last_sell , 'wealth_ma'] = wealth_ma
    buy_p = buy_sell_detail['Buy_Price'][i]
    units_p = wealth_ma/buy_p
    start_i=buy_sell_detail['Buy_Date'][i]
    end_i = buy_sell_detail['Sell_Date'][i]
    wealth_x = wealth_ma
    for x in df_signal[start_i:end_i].index:
        wealth_x = df_signal[df_signal.index == x]['Closing price'] * units_p      
        df_signal.loc[df_signal.index == x , 'wealth_ma'] = wealth_x[0]
    wealth_ma = wealth_x[0]
    
#prefilling the last wealth accumulated till the end of period
last_sell = buy_sell_detail['Sell_Date'][num_buy_sell-1]
df_signal.loc[df_signal.index > last_sell , 'wealth_ma'] = wealth_ma


# In[40]:


#Creatiny BUY and HOLD strategy
#buy and hold
price_buy = total_df['Closing price'][0]
units = initial_wealth/price_buy
len_period = len(df_signal)
df_signal = df_signal.assign(wealth_hold = np.zeros(len_total))
df_signal['wealth_hold'] = units*total_df['Closing price']
final_wealth = df_signal['wealth_hold'][len_period-1]
print(f'The wealth accumulated by investing ${initial_wealth} and holding is ${final_wealth:.2f}.')
print(f'The wealth accumulated by investing ${initial_wealth} and using the moving avaerge buy/sell strategy is ${wealth_ma:.2f}.')


# ##  Plotting the outcome of the trader

# In[42]:


#Plotting the result
plt.clf()
df_signal[['wealth_ma','wealth_hold']].plot()
plt.show()

