'''

Artificial Neural Networks for Paris Endoscopic Classification of Superficial Neoplastic Lesions. 
Proposal of a standardized implementation pipeline. 

Code written by Stefano Magni. 
If you have questions, please email me at: stefano.magni@outlook.com

Training pipeline that uses the selected model and performs cross validation or Hyperparameters tuning 
Refer to the configuration file in PolyPackage to review all hyperparameters that have been set


'''
from PolyPackage import config
from PolyPackage import utils
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kerastuner as kt
import os
import json
from scipy import interp
from time import gmtime, strftime
from sklearn.utils import class_weight
import cv2
SEED = 1234
np.random.seed(SEED)


labels_paris = {
    'LST':3,
    'flat':2, 
    'peduncolate':1, 
    'sessile':0     
}
labels_nice = {
    'adenomatous':0,
    'hyperplastic':1, 
    'serrated':2    
}

def train(train_df, test_df, class_type, network):
    
    # create folders 
    print("[INFO] Creating directories for training output ...")
    print(str(config.CHECKPOINT_DIR))
    if not os.path.isdir(str(config.CHECKPOINT_DIR)):
        
        os.makedirs(str(config.CHECKPOINT_DIR))          
    if not os.path.isdir(str(config.TF_LOGS)):
        os.makedirs(str(config.TF_LOGS))
        
    save_dir = str(config.AUGMENT_DIR)
    if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    if not os.path.isdir(config.OUTPUT_PATH):
            os.makedirs(config.OUTPUT_PATH)
            

    print(f"[INFO] Performing {config.KFOLD} -fold-cross validation ...")
    print(f"Working with the entire dataset of length: {len(train_df)}")
    X = train_df.filenames_dest
    y = train_df.label
    skf = StratifiedKFold(n_splits=config.KFOLD)
    skf.get_n_splits(X, y)
    
    fold = 0
    model_list = []
    metrics_results_dict = {}
    A_matrix = []
    tpr_folds = []
    fpr_folds = []
    auc_folds = []
    for train_index, val_index in skf.split(X, y):
        print(f"[INFO] Training in fold {fold +1} ... ")
        fold = fold + 1

        train_data = train_df[train_df.index.isin(train_index)]
        valid_data = train_df[train_df.index.isin(val_index)]
        
        
        print("[INFO] Constructing Data Generators ...")
        #print(str(train_df['filenames_dest'][3][1]))
        mask_frame = cv2.imread(str(train_df.iloc[2, 0]), cv2.IMREAD_GRAYSCALE)
        print(config.INPUT_SHAPE[:2])
        resized_frame = cv2.resize(mask_frame, (config.INPUT_SHAPE[1], config.INPUT_SHAPE[0]), interpolation = cv2.INTER_AREA)
        cv2.imwrite(config.LOGS_DIR+'/resized_mask_frame.png', resized_frame)
  
        print("[INFO] Preparing Train generator ...")
        train_gen = PolypDataGen(train_data, batch_size = config.BATCH_SIZE, masking_img = resized_frame, threshold = 5, net = network, logs_dir = config.AUGMENT_DIR, augment = config.AUGMENT_DATA)
        print("[INFO] Preparing Valid generator ...")
        valid_gen = PolypDataGen(valid_data, batch_size = config.BATCH_SIZE, masking_img = resized_frame, threshold = 5, net = network, logs_dir = config.AUGMENT_DIR, augment = False, shuffle = False)
        
        if config.DCAPS_TUNE and config.RESNET_TUNE: 
            if network == 'caps': 
                from PolyPackage.dcapsModel import model_build
                
                print("[INFO] Instantiating a bayesian optimization tuner object for D-Caps Network...")
                es = [EarlyStopping(monitor="val_out_caps_accuracy", min_delta=0, patience=8, verbose=0, mode='min')]
                tuner = kt.BayesianOptimization(
                    model_build,
                    objective=kt.Objective("val_out_caps_accuracy", direction="min"),
                    max_trials=10,
                    seed=SEED,
                    directory=config.OUTPUT_PATH,
                    project_name='Bayesian Optimization D-Caps Network')
                
                
            elif network == 'resnet50':
                from PolyPackage.resnet50Model import model_build 
                print("[INFO] Instantiating a bayesian optimization tuner object for Resnet50 Network...")
                es = [EarlyStopping(monitor="val_accuracy", min_delta=0, patience=8, verbose=0, mode='min')]
                tuner = kt.BayesianOptimization(
                    model_build,
                    objective="val_accuracy",
                    max_trials=10,
                    seed=SEED,
                    directory=config.OUTPUT_PATH+'/'+str(class_type),
                    project_name='Bayesian Optimization Resnet50')
            else: 
                print("[INFO] Impossible to find specified model. Check it is implemented in train.py")
             
             
            print("[INFO] Performing hyperparameter search...") 
            tuner.search(
                train_gen, 
                steps_per_epoch = len(train_gen), 
                validation_data=valid_gen,
                validation_steps = len(valid_gen), 
                callbacks= es,
                epochs=config.EPOCHS
            )   
            
            bestHP = tuner.get_best_hyperparameters(num_trials=10)[0]
            if network == 'caps': 
                # grab the best hyperparameters
                print("[TUNING OUT] Best Hyper-Params for caps network")
                kernel_size = bestHP.get("kernel_size")
                print(f"[TUNING OUT] optimal kernel_size: {kernel_size}")
                opt_lr = bestHP.get("learning_rate")
                print(f"[TUNING OUT] optimal learning rate: {opt_lr}")
            elif network == 'resnet50': 
                
                regularization_par = bestHP.get('regular')
                print(f"[TUNING OUT] optimal value for l1 regularizer: {regularization_par}")
                # grab the best hyperparameters
                if config.WHICH_HEAD == 'custom': 
                    dr_1 = bestHP.get("dropout_1")
                    dr_2 = bestHP.get("dropout_2")
                    num_head_n = bestHP.get("units")
                    print("[TUNING OUT] Best Hyper-Params for Resnet network - custom head")
                    print(f"[TUNING OUT] optimal value for droput 1: {dr_1}")
                    print(f"[TUNING OUT] optimal value for dropout 2: {dr_2}")
                    print(f"[TUNING OUT] optimal value for units: {num_head_n}")
                else: 
                    pool_bst = bestHP.get("pooling_1")
                    num_head_n = bestHP.get("units")
                    print("[TUNING OUT] Best Hyper-Params for Resnet network - default head")
                    print(f"[TUNING OUT] optimal value for units: {num_head_n}")
                    print(f"[TUNING OUT] optimal layer before FC Head: {pool_bst}")
                
                opt_lr = bestHP.get("learning_rate")
                print(f"[TUNING OUT] optimal learning rate: {opt_lr}")
                # if config.FINE_TUNING: 
                #     opt_num_layers = bestHP.get("trainable_layers")
                #     print(f"[TUNING OUT] Optimal number of trainable layers: {opt_num_layers}")
                
            
            print("[INFO] Training the best model...")
            model_train = tuner.hypermodel.build(bestHP)
            
        else: 
            hp = 'nothing'
            if network == 'caps':
                from PolyPackage.dcapsModel import model_build
                model_train = model_build(hp)
            elif network == 'resnet50':
                from PolyPackage.resnet50Model import model_build
                model_train = model_build(hp)
            else: 
                print("[INFO] Impossible to find specified model. Check it is implemented in train.py")
        
        time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        callbacks = get_callbacks(config.LOGS_DIR, config.CHECKPOINT_DIR, config.TF_LOGS, time)
        
        
        if class_type == "paris":
            new_df_labels = train_data.replace({"label": labels_paris})
            y_train =new_df_labels["label"]
            curr_classes = np.array(list(labels_paris.values()))
        elif class_type == "nice": 
            new_df_labels = train_data.replace({"label": labels_nice})
            y_train =new_df_labels["label"]
            curr_classes = np.array(list(labels_nice.values()))
        else: 
            print("[INFO] Cannot perform weight balancing")
        
        print("[INFO] Selected class weights ")
        class_weights = class_weight.compute_class_weight('balanced', classes = curr_classes, y = np.array(y_train))
        class_weights  = dict(zip(np.unique(y_train), class_weights))
        print(class_weights)
        # class_weights  = dict(zip(np.unique(y_train), class_weights))
        H = model_train.fit(
            train_gen, 
            validation_data=valid_gen, 
            batch_size=config.BATCH_SIZE,
            class_weight= None,
            epochs=config.EPOCHS, callbacks=callbacks, verbose=1
        )
        
        time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        output_models_dir = config.LOGS_DIR+'/models/'
        if not os.path.isdir(output_models_dir):
            os.makedirs(output_models_dir)
        model_train.save(output_models_dir+'model_'+str(network)+'_'+str(class_type)+'_'+str(time)+'fold_'+str(fold)+'.h5')
        model_train.save_weights(output_models_dir+'model_weights'+str(network)+'_'+str(class_type)+'_'+str(time)+'fold_'+str(fold)+'.h5', save_format="h5")
        
        model_list.append(model_train)
        print("[INFO] Plotting the results ...")
        
        if network == 'resnet50':
            utils.plot_training(H, config.OUTPUT_PATH, network, time)
            
            metrics_out = {'precision': H.history['precision'], 'recall': H.history['recall'], 'accuracy': H.history['accuracy'], 'auc': H.history['auc'],
                        'val_precision': H.history['val_precision'], 'val_recall': H.history['val_recall'], 'val_accuracy': H.history['val_accuracy'],'val_auc': H.history['val_auc'] }    

            key_dict_fold = 'fold_'+str(fold)
            metrics_results_dict[key_dict_fold] = metrics_out
        elif network == 'caps': 
            utils.plot_training_caps(H, config.OUTPUT_PATH, network, time)
            metrics_out = {'out_caps_precision': H.history['out_caps_precision'], 'out_caps_recall': H.history['out_caps_recall'], 'out_caps_accuracy': H.history['out_caps_accuracy'], 
                        'val_out_caps_precision': H.history['val_out_caps_precision'], 'val_out_caps_recall': H.history['val_out_caps_recall'], 'val_out_caps_accuracy': H.history['val_out_caps_accuracy'] }
            key_dict_fold = 'fold_'+str(fold)
            metrics_results_dict[key_dict_fold] = metrics_out
            
        print("----------------------------")  
        print(f"[RESULTS] Model fold {fold}")  
        for key, value in metrics_out.items():
            print(key, ' : ', value[-1])
        print("----------------------------")
          
        # 1. FROM HERE PREDICT ON CURRENT VALID DATAGEN
        
        
        valid_gen = PolypDataGen(valid_data, batch_size = 1, masking_img = resized_frame, threshold = 5, net = network, logs_dir = config.AUGMENT_DIR, augment = False, shuffle = False)
        proba = model_train.predict(valid_gen, len(valid_gen), verbose = 1)
        y_true = valid_data['label_one_hot']
        
        #### CAPS NETWORK HAS DIFFERENT OUTPUT STRUCTURE (SEE LALONDE 2020)
        if network=='caps': 
            out_proba = proba[0]
            # val_reconstructions = proba[1]
        else: 
            out_proba = proba
            
        ##### FROM PROBA INTO CLASSES
        predictions = []
        # here I transform the probabilitites into classes. 
        # For the purpose of the confusion matrix I allow more than one class
        for i, prob in enumerate(out_proba): 
            binary_pred = np.zeros(len(prob))
            index = np.argmax(prob)
            
            binary_pred[index] = 1
            predictions.append(binary_pred)
        
        
        results_df = {'predictions': list(predictions), 'probabilities': list(out_proba), 'label_one_hot': list(y_true), 'label': list(valid_data["label"])}
        results_df = pd.DataFrame(results_df)
        if not os.path.isdir(config.AUGMENT_DIR+'/results'):
            os.makedirs(config.AUGMENT_DIR+'/results')
            
        results_df.to_csv(config.AUGMENT_DIR+'/results/'+'fold_'+str(fold)+'network_'+ network+'time_'+ str(time) +'class_type_'+str(class_type)+'.csv')
        
        ##### DECODE CLASSES
        if class_type == 'nice':
            labels_dict = {
            'adenomatous':0, 
            'hyperplastic':1, 
            'serrated':2
        }
            labels_direct = {
            0: 'adenomatous', 
            1: 'hyperplastic', 
            2: 'serrated'
        }
        else: 
            labels_dict = {
                'LST':0,
                'flat':1, 
                'peduncolate':2, 
                'sessile':3     
            }
            labels_direct = {
            3:'sessile', 
            2:'peduncolate', 
            1:'flat', 
            0:'LST'
        } 
        # 2. COMPUTE ROC CURVE 
        array_true_labels = np.asarray(list(y_true))
        print(array_true_labels.shape)
        print(array_true_labels)
        print(np.array(predictions).shape)
        roc_auc_classes, fpr_classes, tpr_classes =  utils.compute_roc_multiclass(config.NUM_CLASSES, array_true_labels, np.array(predictions))
        
        _status = utils.plot_roc_multiclass(fpr_classes, tpr_classes, roc_auc_classes, config.NUM_CLASSES, str(fold), network, time, class_type, labels_direct)
        if _status: 
            print(f"[INFO] ROC curves for each class saved correctly")
            print(f"[INFO] TPR: {tpr_classes}")
            print(f"[INFO] FPR: {fpr_classes}")
            print(f"[INFO] AUC: {roc_auc_classes}")
        
        tpr_folds.append(tpr_classes)
        fpr_folds.append(fpr_classes)
        auc_folds.append(roc_auc_classes)
        
        # 3. COMPUTE CONF MATRIX 
        time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        A = utils.compute_modified_confusion_matrix(array_true_labels, np.array(predictions))
        _status = utils.plot_conf_matrix(A, labels_dict, class_type, network, str(fold), time)
        if _status: 
            print(f"[INFO] Confusion Matrix saved correctly")
           
        A_matrix.append(A)

        # 4. APPLY GRADCAM
        if network == 'resnet50':
            if class_type == 'paris': 
                grad_direct = {
                    3:'sessile', 
                    2:'peduncolate', 
                    1:'flat', 
                    0:'LST'
                } 
            else: 
                grad_direct = {0:'adenomatous', 1:'hyperplastic', 'serrated':2} 
            _status = utils.visualize_output_gradcam(model_list[0], valid_data, out_proba, y_true, grad_direct, fold)     
            if _status: 
                print(f"[INFO] Gradcam saved correctly")
               
        K.clear_session()
       
        if config.DCAPS_TUNE or config.RESNET_TUNE: 
            print("[INFO] Tuning is set to true. Training only in one fold due to resource constraints.")
            break
        # define data gen here with the split (Break cha be performed after first round to avoid k fold when grid search is applied)
    
    # COMPUTE MEAN ROC CURVE 
    _status = utils.mean_roc_multiclass(tpr_folds, fpr_folds, auc_folds, config.NUM_CLASSES, network, time, class_type, labels_direct)
    if _status: 
        print(f"[INFO] Average ROC saved correctly")
    # COMPUTE AVERAGE CONF MATRIX 
    average_A = np.mean(A_matrix, axis = 0)
    _status = utils.plot_conf_matrix(average_A, labels_dict, class_type, network, "average", time)
    if _status: 
        print(f"[INFO] Average Confusion Matrix saved correctly")
        
    dir_save_json = config.LOGS_DIR +'/metrics_k_fold_'+network+'.json'
    with open(dir_save_json, 'w') as fp:
        json.dump(metrics_results_dict, fp, indent= 4)
        
    return model_list # tpr_folds, fpr_folds, auc_folds, A_matrix, out_proba, predictions, y_true, labels_dict



# CALLBACKS 
def get_callbacks(logs_dir, check_dir, tf_log_dir, time_instant):
    monitor_name = 'val_loss'

    csv_logger = CSVLogger(os.path.join(logs_dir, 'csvlogger' + '_log_' + time_instant + '.csv'), separator=',')
    # tb = TensorBoard(tf_log_dir, histogram_freq=0)
    # model_checkpoint = ModelCheckpoint(os.path.join(check_dir, 'cks' + '_model_' + time_instant + '.hdf5'),
                                    #    monitor=monitor_name, save_best_only=True, save_weights_only=True,
                                    #    verbose=1, mode='min')
    lr_reducer = ReduceLROnPlateau(monitor=monitor_name, factor=0.05, cooldown=0, patience=10,verbose=1, mode='min')
    early_stopper = EarlyStopping(monitor=monitor_name, min_delta=0, patience=8, verbose=0, mode='min')

    return [csv_logger, lr_reducer, early_stopper] # model_checkpoint, 


class PolypDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df_data, 
                 batch_size,  
                 masking_img, 
                 threshold, 
                 net = 'resnet50', 
                 num_classes = 4, 
                 X_col_name = 'filenames_dest', 
                 y_col_name = 'label_one_hot',
                 augment = True,
                 input_size=(576, 768, 3),
                 logs_dir = './out_aug',
                 shuffle=True):
        
        self.df = df_data.copy()
        self.X_col_name = X_col_name
        self.y_col_name = y_col_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.input_size = input_size
        self.net = net
        self.shuffle = shuffle  
        self.n = len(self.df)
        self.logs_dir = logs_dir
        self.augment = augment
        self.augmentor = ImageDataGenerator(# samplewise_center=True,
                                            #samplewise_std_normalization=True,
                                            brightness_range=[0.8,1.0], 
                                            zoom_range=[0.9,1.0],
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            # shear_range=0.1,
                                            # fill_mode='nearest',
                                            horizontal_flip=True,
                                            vertical_flip=True, 
                                            rescale=1./255
        )
        
        self.threshold = threshold
        self.im_bw = cv2.threshold(masking_img, self.threshold, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite(self.logs_dir+'/frame_image_aug.png', self.im_bw)
        
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path, target_size):
        'Here I should add preprocessing and data augmentation'
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()
        
        # plt.imshow(image_arr.astype("uint8"), interpolation='nearest')
        # plt.show()
        return image_arr
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        
        # the input is the dataframe with the data to be yeld by generator (known as The Batch)
        path_batch = batches[self.X_col_name]
        data_batch = np.asarray([self.__get_input(x, self.input_size) for x in (path_batch)])
        label_batch = list(batches[self.y_col_name].values)
        
        return data_batch, np.asarray(label_batch)
    
    def __getitem__(self, index):
        # the length of batches is trivially the batch size
        # the index is passed during the training process when the generator is called
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches) 
        
        if self.net =='caps':
            for i, x in enumerate(X):
                x2 = np.copy(x)
                x2 = x2 + abs(np.min(x2))
                x2 /= (np.max(x2) + 1e-7)
                x2 = (x2 - 0.5) * 2.
                X[i,...] = x2  

        if self.augment: 
            X_gen = self.augmentor.flow(X, batch_size=self.batch_size, shuffle=False)

            if self.net == 'caps':
                # add_endoscopic_frame(next(X_gen), self.im_bw), y
                return [add_endoscopic_frame(next(X_gen), self.im_bw), y], [y,add_endoscopic_frame(next(X_gen), self.im_bw)]
            else:
                return add_endoscopic_frame(next(X_gen), self.im_bw), y
        else: 
            if self.net == 'caps':
                return [X, y], [y,X]
            else:
                return X, y           
    
    def __len__(self):
        return self.n // self.batch_size

def add_endoscopic_frame(X_gen_next, im_bw): 
    output_array = X_gen_next
    
    # print(output_array.shape)
    mask1 = im_bw.copy()
    for i, img_gen in enumerate(X_gen_next):  
        result1 = img_gen.copy()
        cv2.imwrite(config.AUGMENT_DIR+'/data_aug_no_overlay_'+str(i)+'_.png',cv2.cvtColor(result1*255, cv2.COLOR_RGB2BGR))
        result1[mask1 == 0] = 0
        result1[mask1 != 0] = img_gen[mask1 != 0]
        output_array[i] = result1
        cv2.imwrite(config.AUGMENT_DIR+'/data_aug_overlay'+str(i)+'_.png', cv2.cvtColor(result1*255, cv2.COLOR_RGB2BGR) )
    #print(output_array.shape)
    return output_array


# OLD DATA AUGMENTATION. INFORMATION CONTENT WAS LOST (TOO HEAVY)
# self.augmentor = ImageDataGenerator(# samplewise_center=True,
#                                             #samplewise_std_normalization=True,
#                                             brightness_range=[0.9,1.0], 
#                                             rotation_range=15,
#                                             zoom_range=[0.6,1.0],
#                                             width_shift_range=0.1,
#                                             height_shift_range=0.1,
#                                             # shear_range=0.1,
#                                             # fill_mode='nearest',
#                                             horizontal_flip=True,
#                                             vertical_flip=True, 
#                                             rescale=1./255
#         )