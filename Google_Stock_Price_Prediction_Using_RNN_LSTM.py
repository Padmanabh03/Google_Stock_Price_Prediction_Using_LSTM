#!/usr/bin/env python
# coding: utf-8

# In[99]:


# importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[100]:


# importing dataset
data_train = pd.read_csv('Google-Train.csv')


# In[101]:


data_train.head()


# In[102]:


data_train.describe()


# In[103]:


train_set = data_train.iloc[: , 1:2].values


# In[104]:


print(train_set)


# In[105]:


# Visualizing the training set
plt.plot(train_set)
plt.title('Train Set Visualization')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()


# In[106]:


# feature scaling by normalizing 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
train_set_scaled = sc.fit_transform(train_set)


# In[107]:


print(train_set_scaled)


# In[108]:


# creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(40,4413):
    X_train.append(train_set_scaled[i-40:i,0])
    y_train.append(train_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)


# In[109]:


print(X_train)


# In[110]:


# reshaping
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1], 1))


# # Initializing the RNN

# In[111]:


rnn =  Sequential()
rnn.add(LSTM(units= 128, return_sequences= True, input_shape = (X_train.shape[1],1)))
rnn.add(Dropout(0.2))
rnn.add(LSTM(units= 128, return_sequences= True))
rnn.add(Dropout(0.2))
rnn.add(LSTM(units= 128, return_sequences= True))
rnn.add(Dropout(0.2))
rnn.add(LSTM(units= 128, return_sequences= True))
rnn.add(Dropout(0.2))
rnn.add(LSTM(units= 128))
rnn.add(Dropout(0.2))
rnn.add(Dense(units=1))


# In[112]:


# compling the RNN
rnn.compile(optimizer='adam', loss='mean_squared_error')


# In[113]:


rnn.fit(X_train,y_train, epochs= 100, batch_size =32)


# In[114]:


# Test set
data_test = pd.read_csv("Google-Test.csv")
real_stock_price = data_test.iloc[:, 1:2].values


# In[115]:


# Predicting the stock price
data_total = pd.concat((data_train['Open'],data_test['Open']),axis=0)
inputs = data_total[len(data_total) - len(data_test)-40:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(40,58):
    X_test.append(inputs[i-40:i,0])
X_test =  np.array(X_test)
X_test =  np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = rnn.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[116]:


print(predicted_stock_price)


# In[117]:


# Visualizing the result
plt.plot(real_stock_price, color = 'red' , label = "Real Google Stock Price")
plt.plot(predicted_stock_price, color = 'green', label = "Predicted Google Stock Price")
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# In[ ]:




