
import numpy as np
import pandas as pd
from pathlib import Path

# Set the random seed for reproducibility
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)


# Load the fear and greed sentiment data for Bitcoin
df = pd.read_csv(Path('Data/btc_sentiment.csv'), index_col="date", infer_datetime_format=True, parse_dates=True)
df = df.drop(columns="fng_classification")

# Load the historical closing prices for Bitcoin
df2 = pd.read_csv('Data/btc_historic.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)['Close']
df2 = df2.sort_index()


# Join the data into a single DataFrame
df = df.join(df2, how="inner")



def window_data(df, window, feature_col_number, target_col_number):
    '''
    function for generating a rolling window of data from a pd.Series
    Returns np.array for target and np.array for feature
    '''
    # init empty containers
    X = []
    y = []

    # loop the length of the data, subtract window size to avoid out of index error
    for i in range(len(df) - window - 1):

        # generate data window for the current iteration through the loop
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]

        # append data to containers
        X.append(features)
        y.append(target)

    #return np.array of each container
    return np.array(X), np.array(y).reshape(-1, 1)



# generate dataset for RNN using a 10 day rolling window, closing column (1) is used to generate the feature and target data.
window_size = 10
feature_column = 1
target_column = 1

X, y = window_data(df, window_size, feature_column, target_column)



# Use 70% of the data for training and the remaineder for testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


from sklearn.preprocessing import MinMaxScaler
# Use the MinMaxScaler to scale data between 0 and 1.
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale the target training and testing sets
scaler.fit(y_train)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)


# Reshape the features for the model, add an additional dimension.
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# Build the LSTM model. 
# The return sequences need to be set to True if you are adding additional LSTM layers, but 
# You don't have to do this for the final layer. 
# Note: The dropouts help prevent overfitting
# Note: The input shape is the number of time steps and the number of indicators
# Note: Batching inputs has a different input shape of Samples/TimeSteps/Features
input_layer = 30
h1 = 30
h2 = 30
input_shape = (X_train_scaled.shape[1],1)


model = Sequential()
model.add(LSTM(units=input_layer, activation='tanh', return_sequences=True, input_shape=input_shape))
model.add(Dropout(rate=0.1))
model.add(LSTM(units=h1, activation='tanh', return_sequences=True))
model.add(Dropout(rate=0.1))
model.add(LSTM(units=h2, activation='tanh', return_sequences=True))
model.add(Dropout(rate=0.1))
model.add(Dense(1))


# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")



# Summarize the model
model.summary()



# Train the model
# Use at least 10 epochs
# Do not shuffle the data
# Experiement with the batch size, but a smaller batch size is recommended
history = model.fit(X_train_scaled, y_train,  epochs=10, shuffle=False, batch_size=1, verbose=1)