#!/usr/bin/env python
# coding: utf-8

# # Select CPU/GPU

# In[ ]:


'''
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '-1' # Set to -1 if CPU should be used CPU = -1 , GPU = 0

gpus = tf.config.experimental.list_physical_devices('GPU')
cpus = tf.config.experimental.list_physical_devices('CPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
elif cpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        logical_cpus= tf.config.experimental.list_logical_devices('CPU')
        print(len(cpus), "Physical CPU,", len(logical_cpus), "Logical CPU")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        '''


# ## Import the packages/libraries

# In[1]:


import keras
import tensorflow as tf
from tensorflow.keras import models, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization,Dropout, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
from tensorflow.keras.metrics import mean_absolute_error,mean_squared_error
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read the data into pandas dataframe

# In[2]:


df = pd.read_excel('rand_all.xls',usecols="A:H")
scaler = StandardScaler() 
#scaler = MinMaxScaler() 


# ## Allocate input/output, standardize/normalize and split data 

# In[3]:


data = df.iloc[:,3:9].values
labels = df.iloc[:,0:3].values
scaled_data = scaler.fit_transform(data)
scaled_labels = scaler.fit_transform(labels)
x_data,x_valid,y_data,y_valid= train_test_split(scaled_data,scaled_labels,test_size = 0.2,random_state = 42)
x_train,x_test,y_train,y_test= train_test_split(x_data,y_data,test_size = 0.1,random_state = 42)
kp_train = y_train[:,0]
ki_train = y_train[:,1]
kd_train = y_train[:,2]
kp_test = y_test[:,0]
ki_test = y_test[:,1]
kd_test = y_test[:,2]
kp_valid = y_valid[:,0]
ki_valid = y_valid[:,1]
kd_valid = y_valid[:,2]


# ## Visualize data distribution

# In[ ]:





# In[ ]:





# In[86]:


sns.displot(kp_test)
#sns.histplot(data)
kp_test.std(axis=0)


# ## Prediction plot function

# In[16]:


def chart_regression(pred,y,sort=True):
    #t = pd.DataFrame({'preds':preds,'y':y.flatten()})
    #if sort:
        #t.sort.values(by = ['y'], inplace = True)
    a = plt.plot(y.tolist(),label='expected')
    b = plt.plot(pred.tolist(),label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()


# ## Regression Model for Kp prediction

# In[87]:


kp_model = Sequential([Dense(36, input_shape=(5,), activation='relu',kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0, stddev=1)),
                    Dense(10, activation='relu',kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0, stddev=1)),
                    Dense(1, activation='linear'),
                   ])

#compile model
kp_model.compile(Adam(lr=0.0005),loss='mean_squared_error')

#create callback
path = 'MLP_kp.h5'
#monitor = EarlyStopping(monitor='val_loss',min_delta=1e-3,patience=20,verbose=1,restore_best_weights=True)
checkpoint = ModelCheckpoint(path, monitor="val_loss", verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#model summary
kp_model.summary()


# ## Train the model 

# In[94]:


kp_history = kp_model.fit(x_train, kp_train, validation_data = (x_valid,kp_valid) , callbacks = callbacks_list,                    batch_size =64, epochs = 1000, shuffle = True, verbose = 2)
#LeakyReLU(alpha=0.1)
#tf.nn.log_poisson_loss


# ## Print the training curve

# In[89]:


#print(history1.history.keys())
# "Loss"
plt.plot(kp_history.history['loss'])
plt.plot(kp_history.history['val_loss'])
plt.title('kp model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# ## Visualize the prediction scatter

# In[90]:


kp_preds = kp_model.predict(x_test)
#plot predictions
plt.title('Kp Predictions VS Ground_truth')
plt.legend(['Ground_truth', 'Predictions'], loc='upper left')
plt.scatter(kp_test,kp_preds)
plt.ylabel('Value')
plt.xlabel('samples')
plt.show()


# ## Plot prediction against true values

# In[91]:


chart_regression(kp_preds[0:70],kp_test[0:70],sort=True)


# ## Evaluate model error/precision

# In[92]:


kp_model.evaluate(x_test, kp_test, batch_size=64)


# ## Compute R2 score

# In[93]:


r2_score(kp_test, kp_preds)


# ## Test prediction

# In[ ]:


Theta = 17
Vw = 9.2
Vref = 6.27
Error = -0.304
Control = -13189
y = np.array([[Theta,Vw,Vref,Error,Control]])


# In[ ]:


#Standardized
ys = (y-data.mean(axis=0))/data.std(axis=0)
prediction = model.predict(ys)
output = prediction*labels[:,0].std(axis=0)+labels[:,0].mean(axis=0)
output


# In[ ]:


#Normalized
ys = (y-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
prediction = model.predict(ys)
output = prediction*(labels.max(axis=0)-labels.min(axis=0))+labels.min(axis=0)
output


# ## Save/Load the model

# In[ ]:


model.save('MLP_kp.h5')
#model = models.load_model('MLP_PID_best.h5')


# ## Regression Model for Ki prediction

# In[21]:


ki_model = Sequential([Dense(18, input_shape=(5,), activation='sigmoid',kernel_initializer = 'glorot_normal'),# activity_regularizer = regularizers.l2(0.001)),
                    #BatchNormalization(axis=1),
                    #Dropout(0.2),
                    Dense(9, activation='sigmoid', kernel_initializer = 'glorot_normal'),# activity_regularizer = regularizers.l2(0.001)),
                    #BatchNormalization(axis=1),
                    #Dropout(0.2),
                    #Dense(32, activation='relu'),# activity_regularizer = regularizers.l2(1e-4)),
                    #BatchNormalization(axis=1),
                    #Dropout(0.3),
                    #Dense(9, activation='relu'),
                    Dense(1, activation='linear'),
                   ])

#compile model
ki_model.compile(Adam(lr=0.0001),loss='mean_squared_error')#, metrics = [r2_score])

#create callback
path = 'MLP_ki.h5'

#monitor = EarlyStopping(monitor='val_loss',min_delta=1e-3,patience=20,verbose=1,restore_best_weights=True)
checkpoint = ModelCheckpoint(path, monitor="val_loss", verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#model summary
ki_model.summary()


# ##  Train the model 
# 

# In[22]:


ki_history = ki_model.fit(x_train, ki_train, validation_data = (x_valid,ki_valid) , callbacks = callbacks_list,                    batch_size =32, epochs = 1000, shuffle = True, verbose = 2)
#LeakyReLU(alpha=0.1)
#tf.nn.log_poisson_loss


# ## Print the training curve

# In[70]:


#print(history.history.keys())
# "Loss"
plt.plot(ki_history.history['loss'])
plt.plot(ki_history.history['val_loss'])
plt.title('KI model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# ## Visualize the prediction scatter

# In[51]:


ki_preds = ki_model.predict(x_test)
#preds = model.predict(x_test[:,0])

#plot predictions
plt.title('KI Predictions VS Ground_truth')
plt.legend(['Ground_truth', 'Predictions'], loc='upper left')
plt.scatter(ki_test,ki_preds)
plt.ylabel('Value')
plt.xlabel('samples')
plt.show()


# ## Plot prediction against true values

# In[52]:


chart_regression(ki_preds[0:80],ki_test[0:80],sort=True)


# ## Evaluate model error/precision

# In[53]:


ki_model.evaluate(x_test, ki_test, batch_size=64)


# ## Compute R2 score

# In[54]:


r2_score(ki_test, ki_preds)


# ## Test prediction

# In[ ]:


Theta = 17
Vw = 9.2
Vref = 6.27
Error = -0.304
Control = -13189
y = np.array([[Theta,Vw,Vref,Error,Control]])


# In[ ]:


#Standardized
ys = (y-data.mean(axis=0))/data.std(axis=0)
prediction = model.predict(ys)
output = prediction*labels[:,0].std(axis=0)+labels[:,0].mean(axis=0)
output


# In[ ]:


#Normalized
ys = (y-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
prediction = model.predict(ys)
output = prediction*(labels.max(axis=0)-labels.min(axis=0))+labels.min(axis=0)
output


# ## Save/Load the model

# In[ ]:


model.save('MLP_ki.h5')
#model = models.load_model('MLP_PID_best.h5')


# ## Regression Model for Kd prediction

# In[33]:


kd_model = Sequential([Dense(5, input_shape=(5,), activation='relu', activity_regularizer = regularizers.l2(0.001)),
                    BatchNormalization(axis=1),
                    #Dropout(0.2),
                    Dense(10, activation='relu', activity_regularizer = regularizers.l2(0.001)),
                    BatchNormalization(axis=1),
                    #Dropout(0.2),
                    #Dense(32, activation='relu'),# activity_regularizer = regularizers.l2(1e-4)),
                    #BatchNormalization(axis=1),
                    #Dropout(0.3),
                    #Dense(9, activation='relu'),
                    Dense(1, activation='linear'),
                   ])

#compile model
kd_model.compile(Adam(lr=0.0001),loss='mean_squared_error')#, metrics = [r2_score])

#create callback
path = 'MLP_kd.h5'

#monitor = EarlyStopping(monitor='val_loss',min_delta=1e-3,patience=20,verbose=1,restore_best_weights=True)
checkpoint = ModelCheckpoint(path, monitor="val_loss", verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#model summary
kd_model.summary()


# ##  Train the model 

# In[34]:


kd_history = kd_model.fit(x_train, kd_train, validation_data = (x_valid,kd_valid) , callbacks = callbacks_list,                    batch_size =32, epochs = 1000, shuffle = True, verbose = 2)
#LeakyReLU(alpha=0.1)
#tf.nn.log_poisson_loss


# ## Print the training curve

# In[35]:


#print(history.history.keys())
# "Loss"
plt.plot(kd_history.history['loss'])
plt.plot(kd_history.history['val_loss'])
plt.title('KD model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# ## Visualize the prediction scatter

# In[38]:


kd_preds = kd_model.predict(x_test)
#preds = model.predict(x_test[:,0])

#plot predictions
plt.title('KD Predictions VS Ground_truth')
plt.legend(['Ground_truth', 'Predictions'], loc='upper left')
plt.scatter(kd_test,kd_preds)
plt.ylabel('Value')
plt.xlabel('samples')
plt.show()


# ## Plot prediction against true values

# In[39]:


chart_regression(kd_preds[0:80],kd_test[0:80],sort=True)


# ## Evaluate model error/precision

# In[41]:


kd_model.evaluate(x_test, kd_test, batch_size=64)


# ## Compute R2 score

# In[42]:


r2_score(kd_test, kd_preds)


# ## Test prediction

# In[237]:


Theta = 17
Vw = 9.2
Vref = 6.27
Error = -0.304
Control = -13189
y = np.array([[Theta,Vw,Vref,Error,Control]])


# In[237]:


#Standardized
ys = (y-data.mean(axis=0))/data.std(axis=0)
prediction = model.predict(ys)
output = prediction*labels[:,0].std(axis=0)+labels[:,0].mean(axis=0)
output


# In[ ]:


#Normalized
ys = (y-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
prediction = model.predict(ys)
output = prediction*(labels.max(axis=0)-labels.min(axis=0))+labels.min(axis=0)
output


# ## Save/Load the model

# In[ ]:


model.save('MLP_kd.h5')
#model = models.load_model('MLP_PID_best.h5')


# ## Clear Keras session

# In[61]:


keras.backend.clear_session()


# In[60]:


1/np.sqrt(5)


# In[79]:


sns.displot(tf.random_normal([1000]))


# In[ ]:




