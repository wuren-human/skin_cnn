import pandas as pd
import numpy as np
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.applications import ResNet50

data=pd.read_csv("../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")
data['image_full_name']=data['image_id']+'.jpg'
X=data[['image_full_name','dx','lesion_id']]

data=pd.read_csv("../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")
data['image_full_name']=data['image_id']+'.jpg'
X=data[['image_full_name','dx','lesion_id']]

train=pd.concat([X_train,y_train],axis=1)
val=pd.concat([X_val,y_val],axis=1)
test=pd.concat([X_test,y_test],axis=1)

from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
encoder.fit(val['dx'])
name_as_indexes_train=encoder.transform(val['dx']) 
val['label']=name_as_indexes_train

encoder=LabelEncoder()
encoder.fit(test['dx'])
name_as_indexes_test=encoder.transform(test['dx']) 
test['label']=name_as_indexes_test

encoder=LabelEncoder()
encoder.fit(test['dx'])
name_as_indexes_test=encoder.transform(test['dx']) 
test['label']=name_as_indexes_test

train_data= train_generator.flow_from_dataframe(dataframe=train,x_col="image_full_name",y_col="dx",
                                                batch_size=10,directory="../input/mnist1000-with-one-image-folder/ham1000_images/HAM1000_images",
                                                shuffle=True,class_mode="categorical",target_size=(28,28))
                                                
test_generator=ImageDataGenerator(rescale = 1./255)    
test_data= test_generator.flow_from_dataframe(dataframe=test,x_col="image_full_name",y_col="dx",
                                              directory="../input/mnist1000-with-one-image-folder/ham1000_images/HAM1000_images",
                                              shuffle=False,batch_size=1,class_mode=None,target_size=(28,28)) 
val_data=test_generator.flow_from_dataframe(dataframe=val,x_col="image_full_name",y_col="dx",
                                            directory="../input/mnist1000-with-one-image-folder/ham1000_images/HAM1000_images",
                                            batch_size=10,shuffle=False,class_mode="categorical",target_size=(28,28))
from keras.callbacks import ReduceLROnPlateau
learn_tune = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=.5, min_lr=0.0001)      
from keras.callbacks import ModelCheckpoint
# Save the model with best weights
#checkpointer = ModelCheckpoint('../input/best.hdf5', verbose=1,save_best_only=True)
model= Sequential()
model.add(Conv2D(4, (3, 3), activation='relu', padding='same',input_shape=(28,28,3),name='cov1')
model.add(MaxPooling2D(pool_size = (2, 2)),name='pool1'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same',name='cov2')
model.add(MaxPooling2D(pool_size = (2, 2)),name='pool2'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same',name='cov3')
model.add(Flatten())
model.add(Dense(32,activation='relu'), name='dense')
model.add(Dense(7, activation='softmax'), name='softmax')

model.compile(optimizer=optimizers.adam(lr=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])
model.fit_generator(generator=train_data,
                            steps_per_epoch=train_data.samples//train_data.batch_size,
                            validation_data=val_data,
                            verbose=1,
                            validation_steps=val_data.samples//val_data.batch_size,
                            epochs=40,callbacks=[learn_tune])   
test_data.reset()
predictions = model.predict_generator(test_data, steps=test_data.samples/test_data.batch_size,verbose=1)
y_pred= np.argmax(predictions, axis=1)
from sklearn.metrics import confusion_matrix 
cm= confusion_matrix(name_as_indexes_test,y_pred)                            
print(cm)
