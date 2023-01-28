import pandas as pd 
import tensorflow as tf
from sklearn.model_selection import train_test_split ## Splits data into training and testing

## Define dataset as well as where the data origin is
dataset = pd.read_csv('cancer.csv')
## Define X and Y. Where X is the data set given to the network, and Y is the result associated with that data set
x = dataset.drop(columns=['diagnosis(1=m, 0=b)'])
y = dataset['diagnosis(1=m, 0=b)']
## Sets aside a set of data to test, so as to avoid overfitting ( doing well on data already seen but not working on new data )
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
## select model
model = tf.keras.models.Sequential()
## Adds layers to the network. Sigmoid is used here as the result is a binary
model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
## Train the model
model.fit(x_train, y_train, epochs=800)
## test the trained model
model.evaluate(x_test, y_test)