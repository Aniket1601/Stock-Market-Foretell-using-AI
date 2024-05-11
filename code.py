import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import matplotlib.pyplot as plt
df=pd.read_csv('NIFTY50
STOCK/HDFC.csv',na_values=['null'],index_col='Date',parse_dates=True,infer_datetime_format=Tru
e)
df.head()
features = ['Open', 'High', 'Low', 'Volume']
def scale(df, features):
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df.index)
feature_transform.head()
output_var = pd.DataFrame(df['VWAP'])
import pickle
# Serialize and write the variable to the file
pickle.dump(scaler, open("minmaxscaler.pkl", 'wb'))
return feature_transform, output_var
feature_transform, output_var = scale(df,features)
def time_split_function(feature_transform,output_var):
timesplit= TimeSeriesSplit(n_splits=10) # 90-10%
for train_index, test_index in timesplit.split(feature_transform):
X_train, X_test = feature_transform[:len(train_index)],
feature_transform[len(train_index): (len(train_index)+len(test_index))]
y_train, y_test = output_var[:len(train_index)].values.ravel(),
output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()
trainX =np.array(X_train)
testX =np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])
return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = time_split_function(feature_transform, output_var)
def construct_model(X_train):
lstm = tf.keras.Sequential()
lstm.add(tf.keras.layers.LSTM(64, input_shape=(1, X_train.shape[2]), activation='relu',
return_sequences=False))
lstm.add(tf.keras.layers.Dense(32, activation='relu'))
lstm.add(tf.keras.layers.Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
return lstm
def master_train(model, X_train, y_train):
history=model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)
y_pred= lstm.predict(X_test)
lstm.save('model_LSTM.h5')
plt.plot(y_test, label='True Value')
plt.plot(y_pred, label='LSTM Value')
plt.title('Prediction by LSTM')
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()
