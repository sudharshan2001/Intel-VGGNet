import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D,BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from VGGNet import VGGNet
from cfg import CFG
from glob import glob

folders = glob(CFG.base_dir)
train_path = './intel-image-classification/seg_train/seg_train'
valid_path = './intel-image-classification/seg_test/seg_test'

model=VGGNet.build(width=CFG.width, height=CFG.height , depth=CFG.channel , classes=CFG.num_classes)


TRAIN_DATAGEN = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.1,
                zoom_range = 0.1, 
                horizontal_flip = True
				)

TEST_DATAGEN = ImageDataGenerator(
                rescale = 1.0/255
                )

train_flow = TRAIN_DATAGEN.flow_from_directory(train_path, 
												batch_size = BATCH_SIZE,
												target_size = (CFG.width,CFG.height), 
												class_mode = 'categorical'
												)

test_flow = TEST_DATAGEN.flow_from_directory(valid_path, 
											batch_size = BATCH_SIZE, 
											target_size = (CFG.width,CFG.height),
											 class_mode = 'categorical'
											 )

TRAINING_NUM = train_flow.n 
VALID_NUM = test_flow.n


STEP_SIZE_TRAIN = TRAINING_NUM // CFG.batch_size 
STEP_SIZE_VALID = VALID_NUM // CFG.batch_size

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(CFG.lr, metrics=['categorical_accuracy'])

reduce_lr=ReduceLROnPlateau(monitor='val_categorical_acc',factor=0.1,patience=5, min_delta=1e-3, verbose=1, min_lr=1e-6)

weights=ModelCheckpoint('vgg_model_weights.hdf5',
                       save_best_only=True,
                       monitor='val_categorical_acc',
                       verbose=1,
                       save_weights_only=False
                       )

early_stopping=EarlyStopping(monitor='val_categorical_acc',patience=5,restore_best_weights=True)

history = model.fit(
                    train_flow,
                    validation_data=test_flow,
                    epochs=CFG.EPOCHS,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_steps=STEP_SIZE_VALID,
                    callbacks=[reduce_lr,weights,early_stopping]
                    )

