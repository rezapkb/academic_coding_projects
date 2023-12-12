#!/usr/bin/env python
# coding: utf-8

# # Implementing Convolutional Neural Network on OCR Dataset

# In[71]:


import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import math

import glob
import os
import cv2
import sys


# In[73]:


data = pd.read_csv("letter-recognition.csv",header=0)

labelencoder = LabelEncoder()
data['letter'] = labelencoder.fit_transform(data['letter']) 

data.head(10)


# In[74]:


# We share train data because this "sample_submission.csv" only has one class/label
x = data.iloc[:, 1:]
y = data['letter'].tolist()

# # Select 10000 rows data as a testing dataset
x_test = x.iloc[0:10000, :].values.astype('float32') # all pixel values 
y_test = y[0:10000] # Select label for testing data
x_train = x.iloc[10000:, :].values.astype('float32') # all pixel values 
y_train = y[10000:]


# In[75]:


y_train=np.array(y_train)
y_test=np.array(y_test)


# In[76]:


print('Train data: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test data: X=%s, y=%s' % (x_test.shape, y_test.shape))


# In[77]:


x_train = x_train.reshape((x_train.shape[0],x_train.shape[1], 1))
x_test  = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))


# In[78]:


print('Train data: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test data: X=%s, y=%s' % (x_test.shape, y_test.shape))


# In[79]:


#CHECK SHAPE DATA
print('x train shape : ', x_train.shape)
print('y_train shape : ', y_train.shape)
print('x_test shape : ', x_test.shape)
print('y_test shape : ', y_test.shape)


# In[80]:


y_train


# In[81]:


x_train


# In[83]:


#normalize inputs:
x_train2 = x_train/255 - 0.5
x_test2 = x_test/255 - 0.5

# Convert class labels to one-hot encoded
y_train2 = keras.utils.to_categorical(y_train)
y_test2 = keras.utils.to_categorical(y_test)

letter_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

y_train_indexes = []
for classs in y_train:
    y_train_indexes.append(letter_classes.index(classs))
    
# convert class labels to one-hot encoded:
y_train2 = keras.utils.to_categorical(y_train_indexes)


# In[88]:


#import necessary building blocks:
from statistics import mean
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras.layers import LeakyReLU
from tensorflow.keras import activations


# In[90]:


# model
model=Sequential()
# layers
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(16,1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='softmax'))

model.summary()


# In[91]:


# compiling the model
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.005),loss='categorical_crossentropy',metrics=['accuracy'])


# In[92]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[93]:


#fitting the model
history=model.fit(x_train,y_train,epochs=10, validation_data=(x_test,y_test))


# In[94]:


# Predict using testing data without labels/classes
y_pred_test = model.predict(x_test2)
y_pred_test_classes = np.argmax(y_pred_test, axis=1) # Change to normal classes
y_pred_test_classes


# In[95]:


# Create the same format for actual classes
y_actual_test_classes = np.argmax(y_test2, axis=1) # Change to normal classes
y_actual_test_classes


# In[96]:


from sklearn.metrics import multilabel_confusion_matrix
from math import sqrt

# Actual and predicted classes
lst_actual_class = y_actual_test_classes
lst_predicted_class = y_pred_test_classes

# Class = Label 0 to 9
lst_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

# Compute multi-class confusion matrix
arr_out_matrix = multilabel_confusion_matrix(lst_actual_class, lst_predicted_class, labels=lst_classes)

# Temp store results
store_sens = [];
store_spec = [];
store_acc = [];
store_bal_acc = [];
store_prec = [];
store_fscore = [];
store_mcc = [];
for no_class in range(len(lst_classes)):
    arr_data = arr_out_matrix[no_class];
    print("Print Class: {0}".format(no_class));

    tp = arr_data[1][1]
    fp = arr_data[0][1]
    tn = arr_data[0][0]
    fn = arr_data[1][0]
    
    sensitivity = round(tp/(tp+fn), 3);
    specificity = round(tn/(tn+fp), 3);
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
    balanced_accuracy = round((sensitivity+specificity)/2, 3);
    precision = round(tp/(tp+fp), 3);
    f1Score = round((2*tp/(2*tp + fp + fn)), 3);
    x = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    MCC = round(((tp * tn) - (fp * fn)) / sqrt(x), 3)
    store_sens.append(sensitivity);
    store_spec.append(specificity);
    store_acc.append(accuracy);
    store_bal_acc.append(balanced_accuracy);
    store_prec.append(precision);
    store_fscore.append(f1Score);
    store_mcc.append(MCC);
    print("TP={0}, FP={1}, TN={2}, FN={3}".format(tp, fp, tn, fn));
    print("Sensitivity: {0}".format(sensitivity));
    print("Specificity: {0}".format(specificity));
    print("Accuracy: {0}".format(accuracy));
    print("Balanced Accuracy: {0}".format(balanced_accuracy));
    print("Precision: {0}".format(precision));
    print("F1-Score: {0}".format(f1Score));
    print("MCC: {0}\n".format(MCC));


# In[97]:


print("Overall Performance Prediction:");
print("Sensitivity: {0}%".format(round(mean(store_sens)*100, 4)));
print("Specificity: {0}%".format(round(mean(store_spec)*100, 4)));
print("Accuracy: {0}%".format(round(mean(store_acc)*100, 4)));
print("Balanced Accuracy: {0}%".format(round(mean(store_bal_acc)*100, 4)));
print("Precision: {0}%".format(round(mean(store_prec)*100, 4)));
print("F1-Score: {0}%".format(round(mean(store_fscore)*100, 4)))
print("MCC: {0}\n".format(round(mean(store_mcc), 4)))


# # The result of the Best Achieved accuracy seems very promisng, yet we go further to compare the results in details with WEKA results.

# # MAE

# In[98]:


import scipy
import numpy
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_actual_test_classes , y_pred_test_classes)
  




# In[100]:


import math
 
MSE = np.square(np.subtract(y_actual_test_classes, y_pred_test_classes)).mean() 
 
RMSE = math.sqrt(MSE)
RMSE


# # The results from Convolutional Neural Network seems inconsistent! 
# 
# # While the accuracy seems extremely great, the mean absolute error and RMSE values are so bad that we can conclude it is not a good choice to treat the problem.

# In[ ]:




