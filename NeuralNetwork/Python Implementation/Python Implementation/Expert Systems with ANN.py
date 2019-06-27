#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the required libraries

import tensorflow as tf
import numpy as np
import pandas as pd

tf.__version__


# In[4]:


#importing the necessary data
## reading data from excel file ##
data = pd.read_excel('data_new.xlsx')

data.shape
#type(data)


# In[38]:


#splitting the training and testing data on your own

x_train = data.iloc[1:451, 0:10]
x_train= x_train.values

x_train= np.append(x_train, data.iloc[-350:, 0:10].values, axis=0)

x_train.shape

# x_train.transpose().shape

# x_actual_train = x_train.transpose()
# x_actual_train.shape

y_train= data.iloc[1:451, 11:12].values
y_train= np.append(y_train, data.iloc[-350:, 11:12].values, axis=0)

# y_train=y_train.values


##creating testing data set


x_test = data.iloc[451:675, 0:10].values
y_test = data.iloc[451:675, 11:12].values


print("Shapes, of training ", x_train.shape, y_train.shape)
print("Shapes, of testing ", x_test.shape, y_test.shape)


# In[18]:


#splitting the validation data from the fit function

x_data_full = data.iloc[1:, 0:10].values
y_data_full = data.iloc[1:, 11:12].values

print("Shapes of input and output matrix vectors : ", x_data_full.shape, y_data_full.shape)


# In[9]:


#building the model :

#( you can run this block of code again and again 
# to reintitialize the weights and matrices)

#building with one hidden layer and starting off with ten neurons

model_for_training = tf.keras.models.Sequential()
model_for_training.add(tf.keras.layers.Flatten())
model_for_training.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model_for_training.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))


# In[11]:


#specifying and training to build the model

model_for_training.compile(optimizer= 'adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

history_of_training=model_for_training.fit(x_data_full, y_data_full, validation_split=0.20, epochs=80)

#setting the epochs as 80


# In[14]:


import matplotlib.pyplot as plt

#plotting graphs:
#import matplotlib as plt

#plotting training and validation accuracy values
plt.plot(history_of_training.history['acc'])
plt.plot(history_of_training.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#plotting training and validation loss values

plt.plot(history_of_training.history['loss'])
plt.plot(history_of_training.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')


# In[16]:


#now building the model with 15 neurons in the hidden layer and increasing the epochs to upto 90


model_for_training = tf.keras.models.Sequential()
model_for_training.add(tf.keras.layers.Flatten())
model_for_training.add(tf.keras.layers.Dense(15, activation=tf.nn.relu))
model_for_training.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))


#compiling the model by increamenting the epochs to 90


#specifying and training to build the model

model_for_training.compile(optimizer= 'adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

history_of_training=model_for_training.fit(x_data_full, y_data_full, validation_split=0.20, epochs=90)


# In[17]:


## visualizing the the graph plots ##

#plotting training and validation accuracy values
plt.plot(history_of_training.history['acc'])
plt.plot(history_of_training.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#plotting training and validation loss values

plt.plot(history_of_training.history['loss'])
plt.plot(history_of_training.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')


# In[34]:


# now finally incrementing the epochs to about 198 and neurons 25 checking out the visualisation
model_for_training = tf.keras.models.Sequential()
model_for_training.add(tf.keras.layers.Flatten())
model_for_training.add(tf.keras.layers.Dense(25, activation=tf.nn.relu))
model_for_training.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))


#compiling the model by increamenting the epochs to 98


#specifying and training to build the model

model_for_training.compile(optimizer= 'adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

history_of_training=model_for_training.fit(x_data_full, y_data_full, validation_split=0.20, epochs=198)


# In[35]:


## visualizing the the graph plots ##

#plotting training and validation accuracy values
plt.plot(history_of_training.history['acc'])
plt.plot(history_of_training.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#plotting training and validation loss values

plt.plot(history_of_training.history['loss'])
plt.plot(history_of_training.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')


# In[36]:


#building a multi layered perceptron with first hidden layer : 25 neurons second hidden layer 35 neurons


model_for_training = tf.keras.models.Sequential()
model_for_training.add(tf.keras.layers.Flatten())
model_for_training.add(tf.keras.layers.Dense(25, activation=tf.nn.relu))
model_for_training.add(tf.keras.layers.Dense(35, activation=tf.nn.relu))
model_for_training.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))


#compiling the model by increamenting the epochs to 198


#specifying and training to build the model

model_for_training.compile(optimizer= 'adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

history_of_training=model_for_training.fit(x_data_full, y_data_full, validation_split=0.20, epochs=198)


# In[37]:


## visualizing the the graph plots ##

#plotting training and validation accuracy values
plt.plot(history_of_training.history['acc'])
plt.plot(history_of_training.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#plotting training and validation loss values

plt.plot(history_of_training.history['loss'])
plt.plot(history_of_training.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')


# In[45]:


# working with our custom split data
#single hidden layer with 15 neurons 

model_for_training = tf.keras.models.Sequential()
model_for_training.add(tf.keras.layers.Flatten())
model_for_training.add(tf.keras.layers.Dense(15, activation=tf.nn.relu))
#model_for_training.add(tf.keras.layers.Dense(5, activation=tf.nn.relu))
model_for_training.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))



model_for_training.compile(optimizer= 'adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

history_of_training=model_for_training.fit(x_train, y_train, epochs=98)


# In[41]:


predictions = model_for_training.predict(x_test)
print("The predictions for the test matrix is as follows ! ")
predictions


# In[46]:


#evaluating the percent of accuracy for the testingd data
val_loss_for_testing, val_accuracy_for_testing = model_for_training.evaluate(x_test, y_test)

print("Testing loss and accuracy",val_loss_for_testing, val_accuracy_for_testing)


# In[47]:


#plotting graphs:
#import matplotlib as plt

#plotting training and validation accuracy values
plt.plot(history_of_training.history['acc'])
#plt.plot(testing.history['acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#plotting training and validation loss values

plt.plot(history_of_training.history['loss'])
#plt.plot(testing.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')


# In[ ]:




