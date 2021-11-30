'''
Artificial Neural Networks for Paris Endoscopic Classification of Superficial Neoplastic Lesions. 
Proposal of a standardized implementation pipeline. 

thesis objective:
- define a Deep Learning pipeline that can fit within the Operative Unit Activity
- Build a dataset starting from raw clinical data
- experiment with current models and check if they can be extended to multiclass classification. 


Credits: 
- pyImageSearch 
- Rodney Lalonde


Code written by: Stefano Magni
If you have questions, please email me at: stefano.magni@outlook.com

This file handles all the steps required to reproduce the steps to train and test models presented in the thesis
User can set hyperparameters in config.py iside PolyPackage 
User can define dataset directory and properties by passing arguments when calling the script
'''

####### IMPORT REQUIRED LIBRARIES ####### 
import argparse
from PolyPackage import config
from data_preparation import dataset_generation, flow_from_df
from tensorflow.keras import backend as K
from train import train
from test import test
import tensorflow as tf
from PolyPackage.capsule_layers import *

tf.keras.backend.clear_session()
K.clear_session() 

from tensorflow.python.framework.config import set_memory_growth

custom_config=tf.compat.v1.ConfigProto(log_device_placement=True, allow_soft_placement=True)
custom_config.gpu_options.allocator_type = 'BFC'
custom_config.gpu_options.per_process_gpu_memory_fraction = 0.90


# Code works only in experimental mode. 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

    
##### MAIN FUNCTION ###### 
def main(args): 
    """
    Main function that runs the entire experiment

    Args:
        args ([argparse]): inputs from terminal of the user. The Required inputs are the directories of the dataset. Check the arguments in main.py
    """

    if args.extract_ds == 'True': 
        print("[INFO] Generating the dataset from source directory ...")
        train_df, test_df, data_dir_train, data_dir_test = dataset_generation(args.data_dir_excel, args.sequence_dir_csv, args.logs_dir, args.output_dir_dataset, args.dir_converted_videoset, args.dir_cropped_videoset, 
                                           args.num_sequences, args.light_ch, args.img_or_seq, args.sampling_rate, args.class_type, GENIALCO_dir= args.GENIALCO_dir)

    else: 
        print("[INFO] Dataset Generation skipped, loading from file ...")
        train_df, test_df = flow_from_df([], args.output_dir_dataset, True, args.img_or_seq, args.light_ch, args.class_type)
        
    print(f"[INFO] The training samples are: {len(train_df)}")
    print(f"[INFO] The test samples are: {len(test_df)}")
    print(f"[INFO] The sum of train and test: {len(train_df)+ len(test_df) }")

    model_folds = train(train_df, test_df, class_type = args.class_type, network = args.model)
    print(f"[INFO] The number of models saved is: {len(model_folds)}")
    
    print("[INFO] Starting model evaluation ...")
    
    # TESTING PHASE IMPLEMENTED BUT NOT YET PERFORMED 
    # metrics = test(model_folds[0], class_type = args.class_type, model_name = args.model, test_dataframe = test_df)
    
    
    return 1


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Polyp Classification in Fondazione Poliambulanza')
    
    
    # Arguments for the dataset generation part
    parser.add_argument('--data_dir_excel', type=str, required= True, 
                        help= 'directory with the excel file containing structured dataset information')
    parser.add_argument('--sequence_dir_csv', type= str, required=True, 
                        help = 'directory of list of identified sequences in csv format')
    parser.add_argument('--logs_dir', type=str, default='./out_logs', 
                        help='directory where to save output logs')
    parser.add_argument('--output_dir_dataset', type=str, default='./dataset_output/data', 
                        help= 'directory where to save the dataset')
    parser.add_argument('--dir_converted_videoset', type=str, default='./dataset_output/converted', 
                        help= 'directory where to save converted mp4 videos')
    parser.add_argument('--dir_cropped_videoset', type=str, default='./dataset_output/clipped', 
                        help= 'directory where to save clipped mp4 videos')
    parser.add_argument('--num_sequences', type=str, default='all', choices=["all", "one"], 
                        help= 'string to define number of sequences per lesion and light condition. Options: all, one')
    parser.add_argument('--light_ch', type=str, default='both', choices=["W", "NBI", "both"], 
                        help= 'Filter dataset by keeping only White Light, NBI or both')
    parser.add_argument('--img_or_seq', type=str, default='image', choices=["image", "sequence"], 
                        help= 'Select which type of experiment to perform: image based or sequence based. Be sure to use proper model.')
    parser.add_argument('--sampling_rate', type=int, default=15, 
                        help= 'If sequence based dataset is selected with sampling rate you can define at which rate to sample the video. Default is 15')    
    parser.add_argument('--class_type', type=str, default='paris', choices=["paris", "nice"], 
                        help= 'Define which are the target classes for the experiment. Paris: sessile, peduncolate, flat, LST. nice: adenomatous, serrated, hyperplastic, else. Check documentation for further details')
    parser.add_argument('--GENIALCO_dir', type=str, default='./dataset/SOF/', 
                        help= 'dataset where information on intervention data reside. Default: "./dataset/SOF/"')
    parser.add_argument('--extract_ds', type=str, default = 'True')
    parser.add_argument('--model', type=str, default = 'resnet50')
    parser.add_argument('--model_load', type=str, default = 'False')
    
    
    # parser.add_argument('', type=, default='', choices=[], 
    #                     help= '')
    # parser.add_argument('', type=, default='', choices=[], 
    #                     help= '')  
                
    arguments = parser.parse_args()
    main(arguments)



