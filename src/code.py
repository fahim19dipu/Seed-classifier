# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 16:38:33 2021

@author: fahim
"""


import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import keras
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.layers import Dense,Flatten
from keras.initializers import glorot_uniform
from sklearn.metrics import classification_report, confusion_matrix
import my_resnet
#%%
train_path="F:/projects/Seed Classfication with ResNet50/train"
test_path ="F:/projects/Seed Classfication with ResNet50/test"
class_names=os.listdir(train_path)
class_names_test=os.listdir(test_path)
print(class_names)
print(class_names_test)
#%%
train_datagen = ImageDataGenerator(
    
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_path,
     shuffle=True,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_path, # same directory as training data
     shuffle=True,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation') # set as validation data

test_datagen = ImageDataGenerator() 

test_generator = test_datagen.flow_from_directory(
    test_path, # same directory as training data
    target_size=(224, 224),
     shuffle=True,
    batch_size=32,
    class_mode='binary')

#%%


base_model = my_resnet.ResNet50(input_shape=(224, 224, 3))

headModel = base_model.output
headModel = Flatten()(headModel)
headModel=Dense(512, activation='relu', name='fc1',kernel_initializer=glorot_uniform(seed=0))(headModel)
headModel=Dense(256, activation='relu', name='fc2',kernel_initializer=glorot_uniform(seed=0))(headModel)
headModel = Dense( 5,activation='softmax', name='fc3',kernel_initializer=glorot_uniform(seed=0))(headModel)

model = Model(inputs=base_model.input, outputs=headModel)

#model.summary()
#%%
base_model.load_weights("F:/projects/Seed-classifier/src/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")


for layer in base_model.layers:
    layer.trainable = False
    
# for layer in model.layers:
#     print(layer, layer.trainable)


model.compile(keras.optimizers.Adam(learning_rate=0.01),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])    
#%%
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, restore_best_weights=True)
mc = ModelCheckpoint('F:/projects/Seed-classifier/src/models/my_best_model.h5', monitor='val_accuracy', mode='max',save_best_only=True)

H = model.fit_generator(train_generator,validation_data=validation_generator,epochs=30,verbose=1,callbacks=[mc,es])
#%%

modell = keras.models.load_model('my_best_model.h5')
loss, acc = modell.evaluate_generator(validation_generator, steps=3, verbose=0)
print("Validation Accuracy = "+str(acc))
loss, acc = modell.evaluate_generator(test_generator, steps=3, verbose=0)
print("Test Accuracy = "+str(acc))

Y_pred = modell.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

print('Test Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))

print('Test Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=class_names))


model_json = modell.to_json()
with open("F:/projects/Seed-classifier/src/models/model.json","w") as json_file:
  json_file.write(model_json)    