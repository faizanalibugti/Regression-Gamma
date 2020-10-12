#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
import pathlib 

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


print(tf.__version__)


# In[3]:


train_data = np.load('training_data.npy')
df = pd.DataFrame(train_data, columns = ['Yaverage', 'Correct Parameter'])
print(df)


# In[4]:


dataset = df.copy()
dataset.tail()


# In[5]:


train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# In[ ]:


#sns.pairplot(train_dataset[['Yaverage', 'Correct Parameter']], diag_kind='kde')


# In[6]:


train_stats = train_dataset.describe()
train_stats.pop('Correct Parameter')
train_stats = train_stats.transpose()
train_stats


# In[7]:


train_labels = train_dataset.pop('Correct Parameter')
test_labels = test_dataset.pop('Correct Parameter')


# In[8]:


def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# In[9]:


def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(1)
  ])

  model.compile(loss = 'mse',
                optimizer = keras.optimizers.Adam(lr=1e-3),
                metrics = ['mse', 'mae'])
  return model


# In[10]:


model = build_model()


# In[11]:


model.summary()


# In[12]:


example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result


# In[13]:


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epochs, logs):
    if epochs % 100 == 0: print('')
    print('.', end = '')

EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs = EPOCHS, validation_split = 0.2, verbose = 0,
    callbacks = [PrintDot()])


# In[14]:


hist = pd.DataFrame(history.history)
hist['epoch']= history.epoch
hist.tail()


# In[15]:


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 1])
  plt.xlabel('Epoch')
  plt.ylabel('Error [Correct Parameter]')
  plt.legend()
  plt.grid(True)

plot_loss(history)


# In[16]:


model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose = 0, callbacks = [early_stop, PrintDot()])

#plot_history(history)


# In[17]:


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print(mae)


# In[18]:


test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.axis('equal')
plt.axis('square')
plt.xlim(0.45,plt.xlim()[1])
plt.ylim(0.45,plt.ylim()[1])
plt.plot([0.4, 0.6], [0.4, 0.6])


# In[19]:


error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Predicition Error")
plt.ylabel('Count')


# In[20]:


model.save('regression.h5')


# In[23]:


model.predict([0.050216])


# In[24]:


import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('regression.h5')


# In[25]:


model.predict([0.050216])


# In[ ]:




