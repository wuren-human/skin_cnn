# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:13:59 2023

@author: 85734
"""
# import system libs
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from imblearn.over_sampling import RandomOverSampler



model_path = ''
#读取数据
data_dir = 'hmnist_28_28_RGB.csv'
data = pd.read_csv(data_dir)
data.head()
#分类数据和标签
Label = data["label"]
Data = data.drop(columns=["label"])
data["label"].value_counts()
label_counts = data["label"].value_counts()
print(label_counts)

X_total = Data
y_total = Label
#处理不平衡数据
oversample = RandomOverSampler()
Data, Label  = oversample.fit_resample(Data, Label)
Data = np.array(Data).reshape(-1, 28, 28, 3)
print('Shape of Data :', Data.shape)
Label = np.array(Label)
Label
#Convert abbreviations to it's words
classes = {0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'), 
           1:('bcc' , ' basal cell carcinoma'),
           2 :('bkl', 'benign keratosis-like lesions'),
           3: ('df', 'dermatofibroma'),
           4: ('nv', ' melanocytic nevi'),
           5: ('vasc', ' pyogenic granulomas and hemorrhage'),
           6: ('mel', 'melanoma')}



#   train_test_split
X_train , X_test , y_train , y_test = train_test_split(Data , Label , test_size = 0.25 , random_state = 49)

#Create Image Data Generation
datagen = ImageDataGenerator(rescale=(1./255)
                             ,rotation_range=10
                             ,zoom_range = 0.1
                             ,width_shift_range=0.1
                             ,height_shift_range=0.1)

testgen = ImageDataGenerator(rescale=(1./255))

#Create ReduceLROnPlateau to learning rate reduction

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy'
                                            , patience = 2
                                            , verbose=1
                                            ,factor=0.5
                                            , min_lr=0.00001) 


def cnn():
    use_bias = False
    #Model Structure
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=[28, 28, 3]))
    model.add(keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same',name='cov1',use_bias = use_bias))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2),name='pool1'))
    model.add(keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same',name='cov2',use_bias=use_bias))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2),name='pool2'))
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',name='cov3',use_bias = use_bias))
    model.add(keras.layers.Flatten(name='Flatten'))
    model.add(keras.layers.Dense(units=32, activation='relu', name='dense',use_bias = use_bias))
    #要求target为onehot编码
    model.add(keras.layers.Dense(units=7, activation='softmax', name='softmax',use_bias = use_bias))
    return model

# Set a learning rate annealer
if os.path.exists(model_path):
    model = tf.keras.models.load_model(filepath=model_path)
    print(model.summary())
else:
    # 创建 Early Stopping 回调
    model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train ,
                    y_train ,
                    epochs=100,
                    batch_size=10,
                    validation_data=(X_test , y_test) ,
                    callbacks=[learning_rate_reduction,early_stopping])
    #保存模型
    model.save("")
    print(model.summary())

