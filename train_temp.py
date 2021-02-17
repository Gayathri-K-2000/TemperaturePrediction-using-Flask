import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('city_temperature.csv')

df.info()
df.isnull().sum()

df['AvgTemperatureC'] = df['AvgTemperatureC'].fillna(df['AvgTemperatureC'].median())
df['AvgTemperatureF'] = df['AvgTemperatureF'].fillna(df['AvgTemperatureF'].median())
df['Year'] = df['Year'].fillna(df['Year'].median())
df['Month'] = df['Month'].fillna(df['Month'].median())
df['Day'] = df['Day'].fillna(df['Day'].median())

le = LabelEncoder()
df['Country_encoded']=le.fit_transform(df['Country'])
df['City_encoded'] = le.fit_transform(df['City'])
df = df[['Region','Country','City','Country_encoded','City_encoded','Month','Day',	'Year','AvgTemperatureF','AvgTemperatureC']]
df.to_csv('city_temp.csv')
x = df.iloc[:,3:8].values
y = df.iloc[:,9].values

model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

model.compile(loss = tf.losses.MeanSquaredError(), optimizer = tf.optimizers.Adam())

model.fit(x, y, epochs=10)

t_model = 'temperature_model.h5'
model.save(t_model)
                      
