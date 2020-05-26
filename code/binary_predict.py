import os
import sys 
sys.path.insert(0,'F:/Nilgai_photo_database/Nilgai Classifier/tf/code/binary_class')
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input
from binary_datagen import img_crop299
import time



def bi_predict(test_csv, train_csv, my_model_path):

    test_df = pd.read_csv(test_csv) # used for both prediction (without labels), and evaluation (using lables). 
    train_df = pd.read_csv(train_csv) 
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(columns=['Unnamed: 0'])
    else:
        pass

    
    # make sure species ids are strings
    # test_df['OBJECTNAME'] = test_df['OBJECTNAME'].astype(int)
    # train_df['OBJECTNAME'] = train_df['OBJECTNAME'].astype(int)
    # test_df['OBJECTNAME'] = test_df['OBJECTNAME'].astype(str)
    # train_df['OBJECTNAME'] = train_df['OBJECTNAME'].astype(str)
  
    steps = test_df.shape[0]


    test_datagen = ImageDataGenerator(
        rescale = 1./255.,
        preprocessing_function=img_crop299
        )
        
    test_generator = test_datagen.flow_from_dataframe(
        test_df, 
        directory=None,
        x_col='IMGPATH', # test generator for predictions does not have label info
        target_size=(299,299),
        class_mode=None,
        batch_size=1,
        shuffle=False
        )

    train_datagen = ImageDataGenerator(
        rescale = 1./255.,
        # rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=img_crop299 
        )
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        directory=None,
        x_col='IMGPATH',
        y_col='OBJECTNAME',
        target_size=(299,299),
        class_mode='binary',
        batch_size=1,
        shuffle=False,
        seed=1
        )    
    
    # model = tf.keras.models.load_model(my_model_path, compile=True)
    # probability_model = tf.keras.Sequential([model, 
    #                                      tf.keras.layers.Softmax()])

    model = tf.keras.models.load_model(my_model_path, compile=True)
    # probability_model = tf.keras.Sequential([model, 
                                        #  tf.keras.layers.Softmax()])

    pred = model.predict(test_generator, verbose=1, workers=4, steps=steps)

    predicted_class_indices = tf.round(pred).numpy().flatten()
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    conf = pd.DataFrame({'Prediction':list(pred)})
    conf = conf.Prediction.apply(pd.Series)
    conf = conf.max(axis=1)

    # create unique file name from input test_csv file name and with time stamp
    timestr = time.strftime("%Y%m%d-%H%M%S")
    tail = os.path.split(test_csv)[1]
    res_name = timestr + "_" + tail + "_pred_results.csv"

    pred_name = "pred_prob_" + timestr + "_" + res_name
    
    results_dir = 'F:/Nilgai_photo_database/Nilgai Classifier/tf/results'
    pred_results_dir = 'F:/Nilgai_photo_database/Nilgai Classifier/tf/prediciton_results'

    f_name = os.path.join(results_dir, res_name)
    pred_name = os.path.join(pred_results_dir, pred_name)

    # Finally, save the results to a CSV file.
    filenames=test_generator.filenames
    df=pd.DataFrame({"IMGPATH":filenames,
                        "PRED_OBJECTNAME":predictions})

    df['PRED_OBJECTID'] = df['PRED_OBJECTNAME']
    # df['PRED_OBJECTID'] = df['PRED_OBJECTID'].astype(int)

    df['PRED_OBJECTID'] = np.where(df['PRED_OBJECTNAME']=='Nilgai', 1, df['PRED_OBJECTID'] )
    # df['PRED_OBJECTID'] = np.where(df['PRED_OBJECTNAME']=='Animal', 1, df['PRED_OBJECTID'] )
    # df['PRED_OBJECTID'] = np.where(df['PRED_OBJECTNAME']=='None', 0, df['PRED_OBJECTID'] )
    df['PRED_OBJECTID'] = np.where(df['PRED_OBJECTNAME']=='Not_Nilgai', 0, df['PRED_OBJECTID'] )

    df['CONFIDENCE'] = conf

    df_join = df.join(test_df.set_index('IMGPATH'), on='IMGPATH')
    df_join = df_join[['IMGPATH','STUDYAREAN','CONFIDENCE','PRED_OBJECTNAME','OBJECTNAME','PRED_OBJECTID','OBJECT']]
    df_join.columns = ['IMGPATH','STUDYAREAN','CONFIDENCE','PRED_OBJECTNAME','TARG_OBJECTNAME','PRED_OBJECTID','TARG_OBJECT']

    df_join.to_csv(f_name,index=False)
    print("Compare Prediction and Test CSV results saved here:", f_name)
    return pred
  


    