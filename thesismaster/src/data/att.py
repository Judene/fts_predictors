import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from tensorflow.keras.layers import Dense,Dropout,SimpleRNN,LSTM
from tensorflow.keras.models import Sequential
from attention import Attention

# Read input data
input_data = pd.read_csv(r'C:\Users\ELNA SIMONIS\Documents\MEng\2021\Code\Examples\PRSA_data_2010.1.1-2014.12.31.csv')


clean_data = input_data[input_data['pm2.5'].notnull()]
clean_data = clean_data.assign(date_time = pd.to_datetime(clean_data[['year', 'month', 'day', 'hour']]))
clean_data.set_index('date_time', inplace = True)

clean_data['2014'].resample('3D').mean().plot(y = 'pm2.5')
plt.show()

def train_test_split(lookback, clean_data):
    """Splin the input data on train and test subsets.
    """
    y_train = clean_data[:'2013']['pm2.5']
    y_test = clean_data['2014']['pm2.5']

    column_names = ['lag_1']
    x_train = y_train.shift()
    x_test = y_test.shift()
    for i in range(2, 24 * lookback + 1):
        x_train = pd.concat([x_train, y_train.shift(i)], axis = 1)
        x_test = pd.concat([x_test, y_test.shift(i)], axis = 1)
        column_names.append('lag_' + str(i))
    x_train = x_train[24 * lookback:]
    x_test = x_test[24 * lookback:]
    y_train = y_train[24 * lookback:]
    y_test = y_test[24 * lookback:]
    x_train.columns = column_names
    x_test.columns = column_names
    return [x_train, y_train, x_test, y_test]
    
def scale_train_test(x_train, x_test):
    """Normalize train and test subsets
    """
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return [x_train, x_test]

x_train, y_train, x_test, y_test = train_test_split(1, clean_data)

# =============================================================================
# attention
# =============================================================================
lookback = 3
x_train, y_train, x_test, y_test = train_test_split(lookback, clean_data)
x_train, x_test = scale_train_test(x_train, x_test)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

from attention import dot_product, Attention

attention_72 = Sequential()
attention_72.add(LSTM(24, input_shape = (lookback * 24, 1), return_sequences = True))
attention_72.add(Attention())
attention_72.add(Dense(1))
attention_72.summary()

attention_72.compile(loss = 'mean_squared_error', optimizer = 'adam')
#the model was trained during 1500 epochs in total
attention_72.fit(x_train, y_train, epochs = 150, batch_size = 256, verbose = 1)

train_score = attention_72.evaluate(x_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (train_score, math.sqrt(train_score)))
test_score = attention_72.evaluate(x_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (test_score, math.sqrt(test_score)))

plot_forecast(attention_72, x_train, y_train, x_test, y_test)