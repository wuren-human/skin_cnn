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



'''
# 统计训练集中每个类别的图片数量
train_class_counts = [np.sum(np.argmax(y_train, axis=1) == i) for i in range(len(classes))]

# 统计测试集中每个类别的图片数量
test_class_counts = [np.sum(np.argmax(y_test, axis=1) == i) for i in range(len(classes))]

# 类别名称
class_names = [classes[i][0] for i in range(len(classes))]

# 打印每个类别的图片数量
for i, class_name in enumerate(class_names):
    print(f'类别 {class_name} 在训练集中的图片数量：{train_class_counts[i]}, 在测试集中的图片数量：{test_class_counts[i]}')   
    '''
'''
import numpy as np
import pandas as pd
from matplotlib import rcParams

y_true = np.array(y_test)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred , axis=1)
y_true = np.argmax(y_true , axis=1) 
classes_labels = []
for key in classes.keys():
    classes_labels.append(key)

print(classes_labels)

cm = confusion_matrix(y_true, y_pred, labels=classes_labels)
#cm = cm = confusion_matrix(y_true, y_pred, labels=classes_labels)
proportion=[]
for i in cm:
    for j in i:
        temp=j/(np.sum(i))
        proportion.append(temp)
# print(np.sum(confusion_matrix[0]))
#print(proportion)
pshow=[]
for i in proportion:
    pt="%.2f%%" % (i * 100)
    pshow.append(pt)
proportion=np.array(proportion).reshape(7,7)  # reshape(列的长度，行的长度)
pshow=np.array(pshow).reshape(7,7)
#print(pshow)
config = {
    "font.family":'Times New Roman',  # 设置字体类型
}
rcParams.update(config)
plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.PuRd)  #按照像素显示出矩阵
            # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
            # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes,fontsize=12)
plt.yticks(tick_marks, classes,fontsize=12)
 
thresh = cm.max() / 2.
#ij配对，遍历矩阵迭代器
iters = np.reshape([[[i,j] for j in range(7)] for i in range(7)],(cm.size,2))
for i, j in iters:
    if(i==j):
        plt.text(j, i - 0.12, format(cm[i, j]), va='center', ha='center', fontsize=12,color='white',weight=5)  # 显示对应的数字
        plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12,color='white')
    else:
        plt.text(j, i-0.12, format(cm[i, j]),va='center',ha='center',fontsize=12)   #显示对应的数字
        plt.text(j, i+0.12, pshow[i, j], va='center', ha='center', fontsize=12)
 
plt.ylabel('True label',fontsize=16)
plt.xlabel('Predict label',fontsize=16)
plt.tight_layout()
plt.show()

'''

'''
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.show()    
'''
'''
# 获取模型中的所有层
all_layers = model.layers

# 遍历每个层，查找 Dense 层并保存其权重到不同的文件中
dense_layer_count = 0
for layer in all_layers:
    if isinstance(layer, tf.keras.layers.Dense):
        weights = layer.get_weights()
        if len(weights) > 0:
            # 构建文件名，例如：dense_layer_1_weights.txt、dense_layer_2_weights.txt，依此类推
            dense_layer_count += 1
            layer_name = f'dense_layer_{dense_layer_count}'
            filename = f'{layer_name}_weights.txt'
            
            # 保存权重到文本文件
            np.savetxt(filename, weights[0], delimiter='            ', fmt='%.16f')
            
            # 打印文件保存的路径
            print(f"Saved weights of {layer_name} to {filename}")

'''

'''
# 选择一张输入图像（例如，X_test 中的某个样本）
input_image = X_test[0]

# 分别保存每个通道的数据到文本文件
for channel in range(3):
    channel_data = input_image[:, :, channel]
    filename = f'input_image_channel_{channel}.txt'
    np.savetxt(filename, channel_data, delimiter='\n', fmt='%f')
 '''   
    
    
    
    
'''    
middle = Model(inputs=model.input, outputs=model.get_layer('cov1').output)
results = middle.predict(X_test)[0]

# 遍历每个卷积核的输出
for i in range(results.shape[-1]):  # 遍历卷积核的数量
    result = results[:, :, i]  # 提取单个卷积核的输出
    filename = f'D:/tf/FPGA/tensorflow/skin/conv_test/conv/conv1_kernel_{i + 1}.csv'
    np.savetxt(filename, result, delimiter=',')
 
    

# 获取卷积层1的输出
middle = Model(inputs=model.input, outputs=model.get_layer('cov1').output)
result = middle.predict(X_test)[0]
result = result.reshape(result.shape[0], -1)  # 将输出展平成二维数组
np.savetxt('D:/tf/FPGA/tensorflow/skin/conv_test/test/conv1.csv', result, delimiter=',')

# 获取池化层1的输出
middle = Model(inputs=model.input, outputs=model.get_layer('pool1').output)
result = middle.predict(X_test)[0]
result = result.reshape(result.shape[0], -1)
np.savetxt('D:/tf/FPGA/tensorflow/skin/conv_test/test/pool1.csv', result, delimiter=',')

# 获取卷积层2的输出
middle = Model(inputs=model.input, outputs=model.get_layer('cov2').output)
result = middle.predict(X_test)[0]
result = result.reshape(result.shape[0], -1)
np.savetxt('D:/tf/FPGA/tensorflow/skin/conv_test/test/cov2.csv', result, delimiter=',')

# 获取池化层2的输出
middle = Model(inputs=model.input, outputs=model.get_layer('pool2').output)
result = middle.predict(X_test)[0]
result = result.reshape(result.shape[0], -1)
np.savetxt('D:/tf/FPGA/tensorflow/skin/conv_test/test/pool2.csv', result, delimiter=',')

# 获取卷积层3的输出
middle = Model(inputs=model.input, outputs=model.get_layer('cov3').output)
result = middle.predict(X_test)[0]
result = result.reshape(result.shape[0], -1)
np.savetxt('D:/tf/FPGA/tensorflow/skin/conv_test/conv3.csv', result, delimiter=',')

# 获取Flatten层的输出
middle = Model(inputs=model.input, outputs=model.get_layer('Flatten').output)
result = middle.predict(X_test)[0]
result = result.reshape(result.shape[0], -1)
np.savetxt('D:/tf/FPGA/tensorflow/skin/conv_test/Flatten.csv', result, delimiter=',')

# 获取全连接层的输出
middle = Model(inputs=model.input, outputs=model.get_layer('dense').output)
result = middle.predict(X_test)[0]
result = result.reshape(result.shape[0], -1)
np.savetxt('D:/tf/FPGA/tensorflow/skin/conv_test/dense.csv', result, delimiter=',')

# 获取Softmax层的输出
middle = Model(inputs=model.input, outputs=model.get_layer('softmax').output)
result = middle.predict(X_test)[0]
result = result.reshape(result.shape[0], -1)
np.savetxt('D:/tf/FPGA/tensorflow/skin/conv_test/softmax.csv', result,delimiter=',')
'''
'''
# 初始化权重列表
conv_weights = []
#dense_weights = []

# 遍历模型的每一层
for layer in model.layers:
    if isinstance(layer, keras.layers.Conv2D):
        weights = layer.get_weights()
        if weights:
            conv_weights.append(weights)
            
# 保存12，按照 kernel 和 filter 的顺序排序
for i, layer_weights in enumerate(conv_weights):
    for j, kernel_weights in enumerate(layer_weights):
        # 获取卷积核的形状
        kernel_shape = kernel_weights.shape
        # 遍历每个过滤器
        for k in range(kernel_shape[-1]):
            layer_name = f'conv{i + 1}_kernel{j + 1}_filter{k + 1}'  # 获取文件名
            filename = f'{layer_name}_weights.txt'
            # 获取过滤器权重
            filter_weights = kernel_weights[:, :, :, k]
            flattened_weights = filter_weights.flatten()  # 展平权重数组
            np.savetxt(filename, flattened_weights, delimiter=' ', fmt='%.16f')
            '''
'''
# 保存卷积层的权重到不同的文件
for i, weights in enumerate(conv_weights):
    layer_name = f'conv{i + 1}'  # 获取卷积层的名称
    filename = f'{layer_name}_weights.txt'
    flattened_weights = weights[0].flatten()  # 展平权重数组
    np.savetxt(filename, flattened_weights, delimiter=' ', fmt='%.16f')

# 保存全连接层的权重到不同的文件
for i, weights in enumerate(dense_weights):
    layer_name = f'dense{i + 1}'  # 获取全连接层的名称
    filename = f'{layer_name}_weights.txt'
    np.savetxt(filename, weights[0], delimiter=' ', fmt='%.16f')
                
'''
