import os
import sys
sys.path.insert(0,'F:/Nilgai_photo_database/Nilgai Classifier/tf/code/')
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
# save numpy array as csv file
from numpy import savetxt
from categorical_datagen import img_crop



def my_predict(test_csv, train_csv, my_model_path):

    test_df = pd.read_csv(test_csv)
    train_df = pd.read_csv(train_csv) 

    
    # make sure species ids are strings
    # test_df['SPECIESID'] = test_df['SPECIESID'].astype(int)
    # train_df['SPECIESID'] = train_df['SPECIESID'].astype(int)
    test_df['SPECIESID'] = test_df['SPECIESID'].astype(str)
    train_df['SPECIESID'] = train_df['SPECIESID'].astype(str)
    steps = test_df.shape[0]

    test_datagen = ImageDataGenerator(
        rescale = 1./255.,
        preprocessing_function=img_crop
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
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=img_crop 
        )
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        directory=None,
        x_col='IMGPATH',
        y_col='COMMONNAME',
        target_size=(299,299),
        class_mode='categorical',
        batch_size=1,
        shuffle=True,
        seed=1
        )    

    model = tf.keras.models.load_model(my_model_path, compile=True)
    # probability_model = tf.keras.Sequential([model, 
                                        #  tf.keras.layers.Softmax()])

    pred = model.predict(test_generator, verbose=1, workers=4, steps=steps)
    
    predicted_class_indices = np.argmax(pred, axis=1)
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    conf = pd.DataFrame({'Prediction':list(pred)})
    conf = conf.Prediction.apply(pd.Series)
    conf = conf.max(axis=1)

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
    df=pd.DataFrame({
        "IMGPATH":filenames,
        "PRED_COMMONNAME":predictions})

    pred_df = pd.DataFrame(data=pred)
  
    pred_df['IMGPATH'] = filenames


    df['PRED_SPECIESID'] = df['PRED_COMMONNAME']
    # df['PRED_SPECIESID'] = df['PRED_SPECIESID'].astype(int)
    
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='None', '0', df['PRED_SPECIESID'] )
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Pig', '5', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Cattle', '6', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Coyote', '7', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Bobcat', '8', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Humans', '9', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Birds', '10', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Exotics, other', '11', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='White-tailed deer', '12', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Nilgai', '13', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Rabbit', '15', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Raccoon', '17', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Unknown', '18', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Ocelot', '19', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Mouse', '21', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Rat', '22', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Horse', '24', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Dog', '25', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Armadillo', '26', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Opossum', '27', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Spider', '28', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Skunk', '29', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Squirrel', '31', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Tortoise/Turtle', '32', df['PRED_SPECIESID'])
    df['PRED_SPECIESID'] = np.where(df['PRED_COMMONNAME']=='Turkey', '33', df['PRED_SPECIESID'])
    
    df['CONFIDENCE'] = conf
    # df['PRED_PROB'] = pred.tolist()

    df_join = df.join(test_df.set_index('IMGPATH'), on='IMGPATH')
    df_join = df_join[['IMGPATH','STUDYAREAN','CONFIDENCE','PRED_COMMONNAME','COMMONNAME','PRED_SPECIESID','SPECIESID']]
    df_join.columns = ['IMGPATH','STUDYAREAN','CONFIDENCE','PRED_COMMONNAME','TARG_COMMONNAME','PRED_SPECIESID','TARG_SPECIESID']

    df_join = df_join.join(pred_df.set_index('IMGPATH'), on='IMGPATH')

    # df_join['PRED_SPECIESID'] = df_join['PRED_SPECIESID'].astype(int)
    # df_join['PRED_SPECIESID'] = df_join['PRED_SPECIESID'].astype(int)
    # df_join['TARG_SPECIESID'] = df_join['TARG_SPECIESID'].astype(int)
    # df_join['TARG_SPECIESID'] = df_join['TARG_SPECIESID'].astype(int)
    # df_join['PRED_PROB'] = df_join['PRED_PROB'].astype(str)

    # savetxt(pred_name, pred, delimiter=',')    
    # pred = pred.to_numpy()
    # pred = pd.DataFrame({"PRED_PROB":pred, "PRED_LABELS":predictions})
    # pred.to_csv(pred_name, index=False)    
    
    pred_df.to_csv(pred_name, index=False)
    df_join.to_csv(f_name,index=False)
    print("Complete")
    print("Compare Prediction and Test CSV results saved here:", f_name)
    print("Prediction Probabilities saved to...", pred_name)
    return pred
    
    