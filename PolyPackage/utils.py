'''
Artificial Neural Networks for Paris Endoscopic Classification of Superficial Neoplastic Lesions. 
Proposal of a standardized implementation pipeline. 
If you have questions, please email me at: stefano.magni@outlook.com

Code Written by Stefano Magni 14-11-2021. 

Functions to Plot Training History, Confusion Matrices and ROC curves

'''
########### IMPORT REQUIRED LIBRARIES ############
# basic imports
import imutils
import numpy as np
import os 
import cv2

# roc curve
from sklearn.metrics import roc_curve, auc
from scipy import interp # interp tpr

# input image processing for gradcam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# improts from config and GradCAM
from PolyPackage import config
from PolyPackage.gradcam import GradCAM

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns 
from itertools import cycle


########### TRAINING HISTORY ############
def plot_training(training_history, out_dir, network, time):
    """ Function to plot training metrics and loss of a model (resnet).

    Args:
        training_history (tf.keras): Training history returned by model.fit()
        out_dir (str): directory where to save the images
        network (str): which network to save
        time (): time when the model is trained
    """
    
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    f.suptitle(network, fontsize=18)

    ax1.plot(training_history.history['precision'])
    ax1.plot(training_history.history['recall'])
    ax1.plot(training_history.history['accuracy'])
    ax1.plot(training_history.history['val_precision'])
    ax1.plot(training_history.history['val_recall'])
    ax1.plot(training_history.history['val_accuracy'])
    
    ax1.set_title('Precision, Recall, and Accuracy')
    ax1.legend(['Train_Precision', 'Train_Recall', 'Train_Accuracy', 'Val_Precision', 'Val_Recall', 'Val_Accuracy'],
               loc='lower right')
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    ax1.set_xticks(np.arange(0, len(training_history.history['precision'])))
    ax1.grid(True)
    gridlines1 = ax1.get_xgridlines() + ax1.get_ygridlines()
    for line in gridlines1:
        line.set_linestyle('-.')

    ax2.plot(training_history.history['loss'])
    ax2.plot(training_history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(['Train', 'Val'], loc='upper right')
    ax1.set_xticks(np.arange(0, len(training_history.history['loss'])))
    ax2.grid(True)
    gridlines2 = ax2.get_xgridlines() + ax2.get_ygridlines()
    for line in gridlines2:
        line.set_linestyle('-.')

    f.savefig(os.path.join(out_dir, str(network)+ '_plots_' + time + '.png'))
    plt.close()
    
    
def plot_training_caps(training_history, out_dir, network, time):
    """ Function to plot training metrics and loss of a model (capsule).

        Args:
            training_history (tf.keras): Training history returned by model.fit()
            out_dir (str): directory where to save the images
            network (str): which network to save
            time (): time when the model is trained
            
    """
    
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    f.suptitle(network, fontsize=18)

    ax1.plot(training_history.history['out_caps_precision'])
    ax1.plot(training_history.history['out_caps_recall'])
    ax1.plot(training_history.history['out_caps_accuracy'])
    ax1.plot(training_history.history['val_out_caps_precision'])
    ax1.plot(training_history.history['val_out_caps_recall'])
    ax1.plot(training_history.history['val_out_caps_accuracy'])
    
    ax1.set_title('Precision, Recall, and Accuracy')
    ax1.legend(['Train_Precision', 'Train_Recall', 'Train_Accuracy', 'Val_Precision', 'Val_Recall', 'Val_Accuracy'],
               loc='lower right')
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    ax1.set_xticks(np.arange(0, len(training_history.history['val_out_caps_precision'])))
    ax1.grid(True)
    gridlines1 = ax1.get_xgridlines() + ax1.get_ygridlines()
    for line in gridlines1:
        line.set_linestyle('-.')

    ax2.plot(training_history.history['val_out_caps_loss'])
    ax2.plot(training_history.history['val_out_recon_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(['Val Loss', 'Reconstruction Loss'], loc='upper right')
    ax1.set_xticks(np.arange(0, len(training_history.history['loss'])))
    ax2.grid(True)
    gridlines2 = ax2.get_xgridlines() + ax2.get_ygridlines()
    for line in gridlines2:
        line.set_linestyle('-.')

    f.savefig(os.path.join(out_dir, str(network)+ '_plots_' + time + '.png'))
    plt.close()

########### ROC CURVE ############
# DISTINGUISH BETWEEN ARRAY AND PLOT SO THAT I CAN AVERAGE RESULT ON K FOLD IF WANTED


def compute_roc_multiclass(num_classes, y_true, y_pred): 
    # input must be in one hot encoding format extended 
    # roc_curve and auc are imported from sklearn.metrics
    # key of dictionary is the positional encoding of the class in the one hot vector
    """ Compute true positive rate, false positive rate and AUC for each class.

    Args:
        num_classes (int): number of classes present in the classification problem
        y_true (np array): true one hot encoded labels of the test/ validation set
        y_pred (np array): predicted labels after argmax applied to probabilities

    Returns:
        roc_auc: list of AUC for each of the classes (one against the others)
        fpr: false positive rate list for each of the classes (one against the others)
        tpr: true positive rate list for each of the classes (one against the others)
    """
    
    fpr = []
    tpr = []
    roc_auc = []
    for i in range(num_classes):
        current_fpr, current_tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        fpr.append(current_fpr)
        tpr.append(current_tpr)
        roc_auc.append(auc(current_fpr, current_tpr))
    
    return roc_auc, fpr, tpr


def plot_roc_multiclass(fpr, tpr, roc_auc, n_classes, fold, name_model, time, class_type, labels_dict): 
    
    """ Plot the ROC curves for each class given AUC, TPR and FPR for each of them.
    
    Args: 
        fpr: false positive rate for each of the classes
        tpr:  true positive rate for each of the classes
        roc_auc: AUC for each of the classes
        n_classes: number of classes
        fold: index of fold (when cross validation, otherwise use "average" or "single_run")
        name_model: name of the model used
        time: time string when the model is trained
        class_type: experiment performed, either "paris" or "nice"
        labels_dict: dictionary decoding the class from one hot encoding

    Returns:
        plot: one image for each of the classes containing the roc curve
    """
    
    
    start_dir = str(config.AUGMENT_DIR)+'/roc_curve/'
    if not os.path.isdir(start_dir):
        os.makedirs(start_dir)
    colors = cycle(['blue', 'red', 'green', 'orange'])
    for i, color in zip(range(n_classes), colors):
        current_class = labels_dict[i]
        current_auc = roc_auc[i]
        plt.figure(figsize=(15,15))
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=(f'ROC curve of class { current_class} (area = {current_auc:.2f})'))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC for class {current_class}')
        plt.legend(loc="lower right")
    
        dir = start_dir+str(class_type)+'_'+str(name_model)+'_'+fold+'_'+str(time)+'_class_'+str(current_class)+'.png'
        plt.savefig(dir)
        plt.close()
    
    
    return True



def mean_roc_multiclass(tpr_folds, fpr_folds, auc_folds, n_classes, name_model, time, class_type, labels_dict): 
    """ Computes and plots the mean ROC curve over folds for each of the classes.

    Args:
        tpr_folds (np array): false positive rate for each of the classes and folds
        fpr_folds (np array): true positive rate for each of the classes and folds
        auc_folds (np array): AUC for each of the classes and folds
        n_classes (int): number of classes
        name_model (str): name of the model used
        time (timestamp):  time string when the model is trained
        class_type (str): experiment performed, either "paris" or "nice"
        labels_dict (dict): dictionary decoding the class from one hot encoding

    Returns:
        plot: one average ROC curve for each of the classes (over folds)
        
        
    """
    
    # create directory where to save images if non altready present
    start_dir = str(config.AUGMENT_DIR)+'/roc_curve/'
    if not os.path.isdir(start_dir):
        os.makedirs(start_dir)
    
    num_folds = len(tpr_folds)
    mean_fpr = np.linspace(0, 1, 100)
    
    # range over classes
    for i in range(n_classes): 
        current_class = labels_dict[i]
        interp_tpr = []
        # range over folds
        for k in range(num_folds):
            
            interp_tpr.append(interp(mean_fpr, np.array(tpr_folds)[k,i], np.array(fpr_folds)[k,i]))
            interp_tpr[-1][0] = 0.0
        mean_tpr = np.mean(interp_tpr, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(np.array(auc_folds)[:,1])
        
        plt.plot(figsize=(15,15))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
        plt.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(interp_tpr, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC CURVE FOR CLASS {current_class}')
        plt.legend(loc="lower right")
        dir = start_dir+'AVERAGE_ROC_CURVE_'+str(class_type)+'_'+str(name_model)+'_'+str(k)+'_'+str(time)+'_class_'+str(current_class)+'.png'
        plt.savefig(dir)
        plt.close()
    
    return True

######### CONF MATRIX ############
def compute_modified_confusion_matrix(labels, outputs):
    """ Compute a binary multi-class confusion matrix, where the rows
        are the labels and the columns are the outputs
        
    Args:
        labels (array): true one hot encoded labels
        outputs (array): predicted one hot encoded classes

    Returns:
        A: confusion matrix
    """
    num_imgs, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the imgs.
    for i in range(num_imgs):
        # Calculate the number of positive labels and/or outputs.
        # Iterate over all of the classes.
        for j in range(num_classes):
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0

    return A

def plot_conf_matrix(A, labels_dict, class_type, model_name, fold, time): 
    """Given a Matrix computes the heatmap.

    Args:
        A (matrix): Containing outputs vs labels
        labels_dict (dict): dictionary decoding the class from one hot encoding
        class_type (str):  experiment performed, either "paris" or "nice"
        model_name (str): name of the model used
        fold (str): index of fold (when cross validation, otherwise use "average" or "single_run")
        time (timestamp): time string when the model is trained
    Returns:
        plot: confusion matrix 
    """
    start_dir = str(config.AUGMENT_DIR)+'/confusion_matrix/'
    if not os.path.isdir(start_dir):
        os.makedirs(start_dir)
    
    plt.figure(figsize=(15,15))
    h = sns.heatmap(A, annot=True, cmap="Blues")
    h.set_xticklabels(labels=list(labels_dict.keys()), rotation=60)
    h.set_yticklabels(labels=list(labels_dict.keys()), rotation=360)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    dir = start_dir+str(class_type)+'_'+str(model_name)+'_'+fold+'_'+str(time)+'.png'
    plt.savefig(dir)
    plt.close()
    
    return True 
    

def visualize_output_gradcam(model, valid_df, out_proba, y_true, labels_dict, fold):
    """ For ResNet 50 only. Computes the activation map of a chosen layer and displays it. 
        Output image contains confidence for the prediction and heatmap.

    Args:
        model (model): trained model 
        valid_df (dataframe): dataframe containing information on validation/test image dirs
        out_proba (array): output probabilities from the model
        y_true (arrsay): true labels 
        labels_dict (dict):  dictionary decoding the class from one hot encoding
        fold (str): index of fold (when cross validation, otherwise use "average" or "single_run")

    Returns:
        image: input image | heatmap | image + heatmap + predicted class
        
        
    """

    start_dir = config.AUGMENT_DIR +'/gradcam/'+str(fold)
    if not os.path.isdir(start_dir):
        os.makedirs(start_dir)
        
    array_true_labels = np.asarray(list(y_true))
    image_dir = list(valid_df['filenames_dest'])

    for i, prob in enumerate(out_proba): 
        binary_pred = np.zeros(len(prob))
        # the predicted class is where the max probability is found
        index = np.argmax(prob)
        
        binary_pred[index] = 1
        preds = out_proba[i,index]*100
        name_label = labels_dict[index]
        # print(f"[GRADCAM] The predicted class is: {name_label}, with probability {preds}")
        cam = GradCAM(model, index)
        img_dir = image_dir[i]
        image = load_img(img_dir, target_size=config.INPUT_SHAPE)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        heatmap = cam.compute_heatmap(image)
        
        orig = cv2.imread(img_dir)
        resized = cv2.resize(orig, config.INPUT_SHAPE[:2])
        heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

        # superimpose to image the label, the probability over a black rectangle
        cv2.rectangle(output, (0, 0), (430, 65), (0, 0, 0), -1)
        number = format(preds,".2f")
        string_out = f"{name_label}: { number }%"
        cv2.putText(output, string_out, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2)
        output = np.vstack([orig, heatmap, output])
        output = imutils.resize(output, height=1000)
        
        cv2.imwrite(start_dir+'/heatmap_gradcam_'+str(i)+'_'+'.png', output)
    
    return True