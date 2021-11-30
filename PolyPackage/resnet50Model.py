'''

Artificial Neural Networks for Paris Endoscopic Classification of Superficial Neoplastic Lesions. 
Proposal of a standardized implementation pipeline. 

Code written by Stefano Magni. 
If you have questions, please email me at: stefano.magni@outlook.com

Credits: 
How to Add Regularization to Keras Pre-trained Models the Right Way, Silva, Thalles Santos. Nov 26, 2019
https://sthalles.github.io/keras-regularizer/

ResNet 50 Model ready for both Keras Tuning or custom Hyperparameters from Configuration File

'''

############################### IMPORT REQUIRED LIBRARIES ####################################

from PolyPackage import config  # where all the hyperparams are set


#### Layers to define the model ####
import tensorflow as tf
import os
import tempfile
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Flatten, Conv2D, GlobalMaxPool2D, GlobalAvgPool2D, Dropout
from tensorflow.keras import regularizers
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

################################## MODEL COMPILE #########################################
def model_build(hp):
    """Core function that compiles the ResNet 50 model with the chosen parameters

    Args:
        hp (): keras tuner hyperparameters (set to a string when non using it)

    Returns:
        model: compiled Resnet 50 model ready for training
        
    """
    # IF KERAS TUNER IS USED THEN THE INTERVALS FOR THE VARIABLES ARE SET
    if config.RESNET_TUNE: 
        learning_rate = hp.Choice("learning_rate", values= [0.005, 0.0001, 0.0005, 0.00001])
        dropout_1 = hp.Float('dropout_1', min_value=0.0, max_value=0.5, default=0.2, step=0.1) 
        dropout_2 = hp.Float('dropout_2', min_value=0.0, max_value=0.5, default=0.2, step=0.1)
        num_neurons_head = hp.Int('units', min_value=128, max_value=256, step=32, default=128)
        beta_regular = hp.Choice('regular', values=  [0.00,1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]) 
        
    else: 
        learning_rate = config.LEARNING_RATE
        dropout_1 = config.DROPOUT1
        dropout_2 = config.DROPOUT2
        num_neurons_head = config.DENSE_NEURONS_HEAD
        beta_regular = 0.001
        
        hp = 'nothing' # placeholder for hp when non used

    # CONSTRUCT MODEL GRAPH
    custom_model_train = ResNet50(hp, config.INPUT_SHAPE, config.NUM_CLASSES, dropout_1, dropout_2, beta_regular, num_neurons_head)
    
    # Define Training Parameters: 
    # OPTIMIZIER 
    opt = Adam(learning_rate=learning_rate) # beta_1=0.99, beta_2=0.999, decay=1e-6, amsgrad=True)
    
    # METRICS 
    metrics = ['accuracy', tf.keras.metrics.Recall(name = 'recall'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.AUC(name='auc')]
    # LOSS 
    loss = 'categorical_crossentropy' # 'categorical_crossentropy' when multiclass 

    custom_model_train.compile(
        optimizer = opt, 
        loss = loss, 
        metrics = metrics
    )
    
    return  custom_model_train


def ResNet50(hp, input_resnet, number_classes, dropout_1, dropout_2, regular, num_neurons_head): 
    """ Defines the model topology

    Args:
        hp (kt): keras tuner hyperparameters (set to a string when non using it)
        input_resnet (array): size of input images
        number_classes (int): number of classes to be predicted. Should match num classes present in training.
        dropout_1 (int): first dropout layer for the custom head (regularization)
        dropout_2 (int): second dropout layer for the custom head (regularization)
        num_neurons_head (int):Number of neurones in the final dense layer (classification head)

    Returns:
        model: constructed model to be compiled and trained
        
        
    """
    
    # Depending on the input the image net weights are used for Transfer Learning or Fine Tuning 
    if config.TRANSFER_LEARNING: 
        weights = 'imagenet'
        base_resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=weights, 
                                                                            input_shape=input_resnet)
        base_resnet50.trainable = False

        base_resnet50 = add_regularization(base_resnet50, regularizer= tf.keras.regularizers.l2(regular))
        
    elif config.FINE_TUNING:
        weights = 'imagenet'
        base_resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=weights, 
                                                                            input_shape=input_resnet)
        base_resnet50.trainable = True
        base_resnet50 = add_regularization(base_resnet50, regularizer= tf.keras.regularizers.l1_l2(regular, regular))

        # train_layers = hp.Int('trainable_layers', min_value=0, max_value=170, step=10, default=60)
        # up to 177
        train_layers = config.TRAINABLE_LAYERS
        
        for layer in base_resnet50.layers[:train_layers]:
            layer.trainable = False
    else:
        base_resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, 
                                                                            input_shape=input_resnet)
        base_resnet50.trainable = True
        base_resnet50 = add_regularization(base_resnet50, regularizer= tf.keras.regularizers.l2(regular))
    # The head can be either my custom one or the default 
    if config.WHICH_HEAD == 'custom': 
        model = Sequential([
            InputLayer(input_shape=input_resnet),
            base_resnet50,
            Dropout(dropout_1),
            # Classifier
            Flatten(),
            # some regularization is applied
            Dense(num_neurons_head, activation='relu', kernel_initializer = tf.keras.initializers.GlorotUniform()),#kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            Dropout(dropout_2),
            
            # Now for debuggin purposes is set to a binary classification
            # In reality is a num_classes softmax 
            Dense(number_classes, activation='softmax'),
        ])
    else: 
        model = Sequential()

        model.add(InputLayer(input_shape=input_resnet))
        model.add(base_resnet50)
        if config.RESNET_TUNE: 
            if hp.Choice('pooling_1', ['avg', 'max']) == 'max':
                model.add(GlobalMaxPool2D())
            else: 
                model.add(GlobalAvgPool2D()),
        else: 
            model.add(GlobalAvgPool2D()),
        model.add(Dense(num_neurons_head, activation='relu'))
        # Now for debuggin purposes is set to a binary classification
            # In reality is a num_classes softmax
        model.add(Dense(number_classes, activation='softmax'))
        
        model.summary()
        # print(f"the number of layers is {len(model.layers)}")
    
    return model


def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):
    '''
        Credits: 
        How to Add Regularization to Keras Pre-trained Models the Right Way, Silva, Thalles Santos. Nov 26, 2019
        https://sthalles.github.io/keras-regularizer/
        
    '''
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
      print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
      return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)
    
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    
    return model



############## EXPLICIT CONSTRUCTION OF RESNET 50 MODEL ############################
# def hand_ResNet50(input_shape=(64, 64, 3), classes=4):
#     """
#     Implementation of the popular ResNet50 the following architecture:
#     CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
#     -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

#     Arguments:
#     input_shape -- shape of the images of the dataset
#     classes -- integer, number of classes

#     Returns:
#     model -- a Model() instance in Keras
#     """

#     # Define the input as a tensor with shape input_shape
#     X_input = Input(input_shape)

#     # Zero-Padding
#     X = ZeroPadding2D((3, 3))(X_input)

#     # Stage 1
#     X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name='bn_conv1')(X)
#     X = Activation('relu')(X)
#     X = MaxPooling2D((3, 3), strides=(2, 2))(X)

#     # Stage 2
#     X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
#     X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
#     X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

#     ### START CODE HERE ###

#     # Stage 3 (≈4 lines)
#     X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
#     X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
#     X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
#     X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

#     # Stage 4 (≈6 lines)
#     X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

#     # Stage 5 (≈3 lines)
#     X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
#     X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
#     X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

#     # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
#     X = AveragePooling2D((2,2), name="avg_pool")(X)

#     ### END CODE HERE ###

#     # output layer
#     X = Flatten()(X)
#     X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
#     # Create model
#     model = Model(inputs = X_input, outputs = X, name='ResNet50')

#     return model




# def identity_block(X, f, filters, stage, block):
#     """
#     Implementation of the identity block as defined in Figure 3
    
#     Arguments:
#     X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     f -- integer, specifying the shape of the middle CONV's window for the main path
#     filters -- python list of integers, defining the number of filters in the CONV layers of the main path
#     stage -- integer, used to name the layers, depending on their position in the network
#     block -- string/character, used to name the layers, depending on their position in the network
    
#     Returns:
#     X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
#     """
    
#     # defining name basis
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
    
#     # Retrieve Filters
#     F1, F2, F3 = filters
    
#     # Save the input value. You'll need this later to add back to the main path. 
#     X_shortcut = X
    
#     # First component of main path
#     X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
#     X = Activation('relu')(X)

    
#     # Second component of main path (≈3 lines)
#     X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
#     X = Activation('relu')(X)

#     # Third component of main path (≈2 lines)
#     X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

#     # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
#     X = Add()([X, X_shortcut])
#     X = Activation('relu')(X)
    
    
#     return X


# def convolutional_block(X, f, filters, stage, block, s = 2):
#     """
#     Implementation of the convolutional block as defined in Figure 4
    
#     Arguments:
#     X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     f -- integer, specifying the shape of the middle CONV's window for the main path
#     filters -- python list of integers, defining the number of filters in the CONV layers of the main path
#     stage -- integer, used to name the layers, depending on their position in the network
#     block -- string/character, used to name the layers, depending on their position in the network
#     s -- Integer, specifying the stride to be used
    
#     Returns:
#     X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
#     """
    
#     # defining name basis
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
    
#     # Retrieve Filters
#     F1, F2, F3 = filters
    
#     # Save the input value
#     X_shortcut = X


#     ##### MAIN PATH #####
#     # First component of main path 
#     X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
#     X = Activation('relu')(X)

#     # Second component of main path (≈3 lines)
#     X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
#     X = Activation('relu')(X)


#     # Third component of main path (≈2 lines)
#     X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


#     ##### SHORTCUT PATH #### (≈2 lines)
#     X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
#                         kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
#     X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

#     # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
#     X = Add()([X, X_shortcut])
#     X = Activation('relu')(X)
    
    
#     return X