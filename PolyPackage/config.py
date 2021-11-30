'''
Artificial Neural Networks for Paris Endoscopic Classification of Superficial Neoplastic Lesions. 
Proposal of a standardized implementation pipeline. 
If you have questions, please email me at: stefano.magni@outlook.com

Code Written by Stefano Magni. 14-11-2021


Configuration file to tune hyperparameters and define output directories

'''


# define the path to our output directory
OUTPUT_PATH = "output"
OUTPUT_DATASET = "dataset"
# initialize the input shape and number of classes
INPUT_SHAPE = (576, 768, 3)
NUM_CLASSES = 3 # with nice set to 3


# define the total number of epochs to train, batch size, and the
# early stopping patience
KFOLD = 5
EPOCHS = 10
BATCH_SIZE = 8 # max BS = 8 for capsnet
EARLY_STOPPING_PATIENCE = 3
LEARNING_RATE = 0.0001
# TRAINING PARAMETERS 
DEBUG =True
AUGMENT_DATA= True
LOGS_DIR = './logs_tr'
CHECKPOINT_DIR = './logs_tr/cks'
TF_LOGS ='./logs_tr/tf'
AUGMENT_DIR='./aug_output'

# HYPERPARAMETERS RESNET50
RESNET_TUNE = False
DROPOUT1 = 0
DROPOUT2 = 0.20
DENSE_NEURONS_HEAD = 128
TRANSFER_LEARNING = False
FINE_TUNING = True
TRAINABLE_LAYERS = 150
WHICH_HEAD = 'default' # other option is 'default' 

# HYPERPARAMS D-CAPS 
DCAPS_TUNE = False
KERNEL = 5 # kernel size
OUT_ATOMS = 16
ROUTINGS1 = 3 
ROUTINGS2 = 3 
RECONSTRUCTION_WEIGHT_LOSS = 0.0005


# CLASS IMBALANCE
UPSAMPLE = True
UPSAMPLE_RATIO = 0.9 # upsampling equal to : num_items in class + num_items majority class * UPSAMPLE_RATIO