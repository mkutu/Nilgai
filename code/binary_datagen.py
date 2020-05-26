import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.image import crop_to_bounding_box
from tensorflow.keras.applications.inception_v3 import preprocess_input

"""
05/14/2020
todo:
create a preprocessing function that randomly horizontally shifts but consistently 
crops them to the desired sizes. Current preprocessing function does crops consistently
but does not apply random augmentation
"""

def img_crop299(img):
    # for cropping
    offset_width = 0
    offset_height = 9
    target_height = 273
    target_width = 299

    # for random width shift
    wrg = 0.5 
    hrg = 0.0
    row_axis =  2
    channel_axis = 1

    new_img = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)

    # new_img = tf.keras.preprocessing.image.random_shift(new_img, wrg, hrg, 
    # row_axis=row_axis, col_axis=0, channel_axis=channel_axis, 
    # fill_mode='mirror')

    # new_img = tf.image.random_flip_up_down(new_img, seed=1)
    # new_img = tf.image.random_contrast(new_img, 0.2, 0.5, seed=1)
    new_img = tf.image.resize(new_img, (target_width,target_width))
    return new_img


def binary_datagen(train_csv, validation_csv, batch_size, target_size):

    # Load img path and label csvs
    train_df = pd.read_csv(train_csv)
    validation_df = pd.read_csv(validation_csv)

    # make sure species ids are strings
    # validation_df['OBJECTNAME'] = validation_df['OBJECTNAME'].astype(int)
    # train_df['OBJECTNAME'] = train_df['OBJECTNAME'].astype(int)
    validation_df['OBJECTNAME'] = validation_df['OBJECTNAME'].astype(str)
    train_df['OBJECTNAME'] = train_df['OBJECTNAME'].astype(str)

    

    train_datagen = ImageDataGenerator(
    rescale = 1./255.,
    # rotation_range=20, # causes logo 
    # width_shift_range=0.1, #causes logo
    # height_shift_range=0.1, # causes logo
    # shear_range=0.1, # causes logo with vertical flip
    # zoom_range=(0.2,0.9),
    horizontal_flip=True,
    # vertical_flip=True,
    # brightness_range=(0.3,1.5), 
    # channel_shift_range=150.0,
    preprocessing_function=img_crop299,

    )

    validation_datagen = ImageDataGenerator(
    rescale = 1./255.,
    preprocessing_function=img_crop299
    )
        


    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        directory=None,
        x_col='IMGPATH',
        y_col='OBJECTNAME',
        target_size=target_size,
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True,
        seed=1
        )

    validation_generator = validation_datagen.flow_from_dataframe(
        validation_df, 
        directory=None,
        x_col='IMGPATH',
        y_col='OBJECTNAME',
        target_size=target_size,
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True,
        seed=1
        ) 

    return train_generator, validation_generator

    