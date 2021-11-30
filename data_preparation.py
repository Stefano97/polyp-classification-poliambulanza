'''

Artificial Neural Networks for Paris Endoscopic Classification of Superficial Neoplastic Lesions. 
Proposal of a standardized implementation pipeline. 

Code written by Stefano Magni. 
If you have questions, please email me at: stefano.magni@outlook.com

Code to Preprocess data and prepare it for the data generators

'''


####### IMPORT REQUIRED LIBRARIES ####### 
import cv2
import json
import os
import datetime
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import resample
from PolyPackage import config

pd.set_option('display.max_columns', None)
SEED = 1234
np.random.seed(SEED)


######### CATEGORICAL ENCODINGS AND COLUMN NAMES ######### 
# Defining dictionary for classes and their encoding

cols_RSCS_new = {
 'ID_RSCS': 'ID_RSCS',
 'RSCS_DIRECTORY':'RSCS_DIRECTORY', 
 'ID_PATIENT':'ID_PATIENT', 
 'OPERATORE':'ENDOSCOPIST', #cat
 'IA':'AI', #cat
 'PREPARAZIONE':'PREPARATION',#cat
 'MODALITà ASSUNZIONE':'INTAKE', #cat
 'CIECO RAGGIUNTO (si/no)':'CECUM', #cat
 'SE NO, SPECIFICARE perché':'WHY_NOT_CECUM',
 'RETRAZIONE (min)':'RETRACTION_MINUTES', 
 'PULIZIA DX':'BBPS_RIGHT', #cat
 'PULIZIA  TX':'BBPS_TRANSV', #cat
 'PULIZIA  SN':'BBPS_LT',  #cat
 'COMPLICANZE (si/no)':'ADVERSE_EVENT', #cat
 'SE SI, SPECIFICARE':'WHAT_ADVERSE', 
 'POLIPI SI/NO': 'POLYPS', #cat
 'numero polipi': 'NUMBER_POLYPS', 
 'ADENOMI SI/NO': 'ADENOMAS', #cat
 'numero adenomi': 'NUMBER_ADENOMAS',
 'SERRATI SI/NO': 'SERRATED', #cat
 'TUMORI (si/no)': 'CRC', #cat
 'FALSI POSITIVI (si/no)': 'FALSE_POSITIVES', #cat
 'SEDE': 'WHERE_FP_1',
 'COSA':'WHAT_FP_1', 
 'SEDE.1': 'WHERE_FP_2',
 'COSA.1':'WHAT_FP_2', 
 'SEDE.2': 'WHERE_FP_3',
 'COSA.2':'WHAT_FP_3',  
} 

cols_PATIENTS_new = {
    'SESSO':'SEX', 
    'Data nascita':'DOB', 
    'ALTEZZA CM':'HEIGHT_CM', 
    'PESO KG':'WEIGHT_KG', 
    'PRIMA RSCS':'FIRST_RSCS',
    'SE CONTROLLO, N COLONSCOPIA':'NUMBER_RSCS',
    'SE CONTROLLO, DATA PREGRESSA COLONSCOPIA (mm/aaaa)':'DATE_LAST_RSCS',
    'SE CONTROLLO, SONO STATI RIMOSSI ADENOMI?':'PAST_ADENOMAS',
    'SE CONTROLLO, SONO STATI RIMOSSI ADENOMI AVANZATI?':'PAST_ADVANCED_ADENOMAS',
    'SINTOMI (oltre SOF)':'SYMPTOMS'
}

YN = {0:'SI', 1:'NO'}
paris_class_dict = {
    1:'sessile', 
    2:'peduncolate', 
    3:'flat', 
    4:'LST'
} 
paris_classes_inv = {
    'sessile':1, 
    'peduncolate':2, 
    'flat':3, 
    'LST':4
} 
sex_dict = {1:'male', 2:'female'}

what_preparation_dict = {
    1: 'plenvu', 
    2: 'moviprep', 
    3: 'selg-sse etc.', 
    4: 'else'
}

when_preparation_dict = {
    1: 'day before', 
    2:'fractional', 
    3:'same day'
}

symptoms = {
    0: 'asymptmatic', 
    1: 'diarrhea', 
    2: 'anemia', 
    3: 'proctorrhagia', 
    4: 'pain',
    5: 'weight loss',
    6: 'constipation', 
    7: 'occlusion'
}

boston_bowel_preparation_scale = {
    0: 'Unsatisfactory', 
    1: 'Poor', 
    2: 'Fair',
    3: 'Good'
}

location_dict = {
    1:'cecum',
    2:'ileocecal valve', 
    3:'ascending colon (right)', 
    4:'hepatic flexure', 
    5:'transverse colon', 
    6:'splenic flexure',
    7:'descending colon (left)', 
    8: 'sigmoid colon',
    9:'rectum'
}

histologic_dict = {
    0: 'ATV-LGD',
    1: 'AT-LGD',
    2: 'AT-HGD',
    3: 'SSAP-ND',
    4: 'SSAP-D+',
    5: 'TSA',
    6: 'Hyperplastic',
    7: 'tumor',
    8: 'juvenile',
    9: 'ATV-HGD',
    10: 'LEIOMIOMA',
    11: 'schwannoma',
    12: 'INFIAM', 
    13: 'tipo amartomatoso'
}
hist_type = {
    1: 'adenomatous', 
    2: 'hyperplastic', 
    3: 'serrated', 
    4: 'else'
}
hist_type_inv = {
    'adenomatous':0, 
    'hyperplastic':1, 
    'serrated':2, 
    'else':3
}

seen_by_dict = {
    1: 'only AI',
    2: 'only OP',
    3: 'both AI and OP'
}

# save the dictionaries for interpretations
categorical_encoding = {
    'paris_classes':paris_class_dict,
    'bowel_preparation_drug':what_preparation_dict,
    'bowel_preparation_timing':when_preparation_dict,
    'symptoms':symptoms,
    'bbps':boston_bowel_preparation_scale,
    'location':location_dict,
    'histology':histologic_dict,
    'hist_type': hist_type, 
    'hist_type_inv': hist_type_inv, 
    'paris_classes_inv': paris_classes_inv,
    'seen_by':seen_by_dict
}

############################################################

def dataset_generation(data_directory_excel, sequence_directory_excel, logs_directory, dir_definitive_dataset, dir_first_videoset, dir_clipped_videoset, num_sequences = "all", light_ch='both', img_or_seq='image', sampling_rate=7, class_type= 'paris', GENIALCO_dir= './dataset/SOF/', encoding_categorical_df= encoding_categorical_df, col_mapping = cols_RSCS_new,): 
    """ Takes input directories and starting excel file and outputs train and test dataframes

    Args:
        data_directory_excel (str): From user input in terminal, directory of excel file
        sequence_directory_excel (str): From user input in terminal, directory of sequences (obtained during the labelling process)
        logs_directory (str): directory where to save meaningful information returned by this function
        dir_definitive_dataset (str): directory where to save the training dataset
        dir_first_videoset (str): directory where to save videoset containing converted videos
        dir_clipped_videoset (str): directory where to save clipped video frames 
        num_sequences (str, optional): whether to take all or one sequences from each polyp. Defaults to "all".
        light_ch (str, optional): whether to take only white or NBI light or both of them. Defaults to 'both'.
        img_or_seq (str, optional): whether to. Defaults to 'image'.
        sampling_rate (int, optional): sampling rate to extract images from videos. Defaults to 7.
        class_type (str, optional): type of experiment to perform. Defaults to 'paris'.
        GENIALCO_dir (str, optional): source directory where data comes from. Defaults to './dataset/SOF/'.
        encoding_categorical_df (dict, optional): Containing mapping to hide endoscopist names from reports and results. Defaults to encoding_categorical_df.
        col_mapping (dict, optional):Dictionary to encode name of dataframe columns. Defaults to cols_RSCS_new.

    Returns:
        train_df, test_df, data_out_train, data_out_test: train and test dataframes with the final directories of the data ready to be loaded
        
        
    """
    # Create useful directories
    # LOGS 
    if not os.path.isdir(logs_directory):
        os.mkdir(logs_directory)
        print('[INFO] Logs directory created!')

    # OUTPUT DATA DIRECTORY
    output_data_directory = dir_definitive_dataset
    if not os.path.isdir(output_data_directory):
        os.makedirs(output_data_directory)
        print('[INFO] Output Data directory created!')

    converted_data_directory = dir_first_videoset
    if not os.path.isdir(converted_data_directory):
        os.makedirs(converted_data_directory)
        print('[INFO] Converted Data directory created!')

    dir_save_json = logs_directory +'/categorical_encoding.json'
    with open(dir_save_json, 'w') as fp:
        json.dump(categorical_encoding, fp, indent= 4)
    
    # IMPORT THE EXCEL TABLE WITH SUMMARIZED DATA 
    table_raw_data = pd.read_excel(data_directory_excel, engine= 'openpyxl')
    rscs_tbl, patient_tbl, polyp_tbl = normalize_database(table_raw_data, GENIALCO_dir, encoding_categorical_df,logs_directory, cols_map= col_mapping)
    seqs_tbl = sequence_table_from_csv(sequence_directory_excel, logs_directory)
    plyp_tbl_set = custom_train_test_split(rscs_tbl, polyp_tbl, class_type, logs_directory)
    
    df_files = labelled_dataset_creation(rscs_tbl, plyp_tbl_set, hist_type, paris_class_dict, seqs_tbl, class_type, logs_directory, dir_first_videoset, dir_clipped_videoset)
    
    if len(df_files) != 0: 
        print("[INFO] successfully created cropped version of the dataset")
    else: 
        print("[INFO] Issue with cropping, exiting")
        return 0 
    
    df_selected_seqs = selecting_dataset_files(df_files,seqs_tbl, num_sequences, light_ch)
    #(output_dir_dataset, df_sel_seqs, img_seq="image", smpl_rate= 1)
    output_dataframe, data_out_train, data_out_test = from_dirs_to_frames(dir_definitive_dataset, df_selected_seqs, class_type, img_or_seq, sampling_rate, encoded = False)
    # If nice then the "else" class is removed from the list. Only data from Serrated, Hyperplastic and adenomatous is kept
    output_dataframe = nice_adaptation(output_dataframe, class_type)
    train_df, test_df = flow_from_df(output_dataframe, dir_definitive_dataset, True, img_or_seq, light_ch, class_type)
    _success = save_tables(dir_definitive_dataset, logs_directory, output_dataframe, light_ch, img_or_seq, class_type, rscs_tbl, patient_tbl, polyp_tbl, seqs_tbl)
    
    if _success: 
        print("[INFO] Database saved correctly")
        return train_df, test_df, data_out_train, data_out_test


# Starting from the output dataset I have to generate the dataframes needed for flow from dataframe.
# train_df / test_df with two columns: |file_dir_with_ext | label (according to current selection)| 

def nice_adaptation(output_dataframe, class_type): 
    
    """ The classes for Nice are reduced from the 11 at disposal to the major three that are clinically relevant.
        Article Source: A comparative study on polyp classification using convolutional neural networks
        Patel K, Li K, Tao K, Wang Q, Bansal A, et al. (2020) A comparative study on polyp classification using convolutional neural networks. PLOS ONE 15(7): e0236452. 
        https://doi.org/10.1371/journal.pone.0236452
    

    Returns:
        output_dataframe: dataframe containing only those entries that belong to the three classes: adenomatous, hyperplastic and serrated
    """
    
    # Inside here if network is applied to nice all labels that are 'else' are removed
    if class_type == 'nice':  
        output_df = output_dataframe[output_dataframe['label'].isin(['adenomatous', 'hyperplastic', 'serrated'])]
    else: 
        output_df = output_dataframe
    return output_df 

def flow_from_df(output_df, dir_def_dataset, one_hot_enc, img_or_seq, light, class_type): 
    """ Function used to load or receive as input a dataframe and return a train and test dataframe with 
        one hot encoded labels and with resampling minority classes. 

        Even if one hot encoding is offered as an option it is required to train the models. 
    Args:
        output_df (dataframe): dataframe obtained from previous preprocessing steps (containing label names and directories )
        dir_def_dataset (str): directory where the definitive dataset for training resides
        one_hot_enc (Bool): True / False indicating whether to perform one hot encoding of targer variable or not
        img_or_seq (str): indicates user choice of which dataset was chosen (sequence or image)
        light (str): indicates user choice of which light was selected
        class_type (str): indicates which label is considered: nice or paris

    Returns:
        train_df_labelled, test_df_labelled: train and test dataframes used by data generators
    """
    
    if len(output_df) != 0: 
        if one_hot_enc: 
            output_df_labelled = output_df.copy()
            output_df_labelled["label_one_hot"] = output_df_labelled["label"].str.get_dummies().values.tolist()
            # train_df_labelled = output_df_labelled.loc[output_df_labelled.train_test_set=='train'].copy()
            train_df_labelled = output_df_labelled.copy()
            train_df_labelled = train_df_labelled[["filenames_dest", "label", "polyp_id","label_one_hot"]]
            test_df_labelled = output_df_labelled.loc[output_df_labelled.train_test_set=='test'].copy()
            test_df_labelled = test_df_labelled[["filenames_dest", "label", "polyp_id", "label_one_hot"]]
        else: 
                train_df_labelled = output_df.loc[output_df.train_test_set=='train'].copy()
                train_df_labelled = train_df_labelled[["filenames_dest", "label", "polyp_id"]]
                test_df_labelled = output_df.loc[output_df.train_test_set=='test'].copy()
                test_df_labelled = test_df_labelled[["filenames_dest", "label", "polyp_id"]]
            
    if len(output_df) == 0: 
        print("[INFO] Loading csv with train-test partition ...")
        
        df_load_dir = dir_def_dataset+'/dataset_'+img_or_seq+'_'+light+'_'+class_type+'.csv'
        print(f"[INFO] The directory is: {df_load_dir}")
        dataframe_loaded = pd.read_csv(df_load_dir) 
        print(f"[INFO]  The length of the loaded dataset is: {len(dataframe_loaded)}") 
        
        
        if one_hot_enc: 
            dataframe_loaded["label_one_hot"] = dataframe_loaded["label"].str.get_dummies().values.tolist()
            # train_df_labelled = dataframe_loaded.loc[dataframe_loaded.train_test_set=='train'].copy()
            train_df_labelled = dataframe_loaded.copy()
            print(f"[INFO] The training samples are:{len(train_df_labelled)} ")
            train_df_labelled = train_df_labelled[["filenames_dest", "label", "polyp_id", "label_one_hot"]]

            test_df_labelled = dataframe_loaded.loc[dataframe_loaded.train_test_set=='test'].copy()
            test_df_labelled = test_df_labelled[["filenames_dest", "label", "polyp_id", "label_one_hot"]]     
            
        else: 
            # train_df_labelled = dataframe_loaded.loc[dataframe_loaded.train_test_set=='train'].copy()
            train_df_labelled = dataframe_loaded.copy()
            print(f"[INFO] The training samples are:{len(train_df_labelled)} ")
            train_df_labelled = train_df_labelled[["filenames_dest", "label", "polyp_id"]]

            test_df_labelled = dataframe_loaded.loc[dataframe_loaded.train_test_set=='test'].copy()
            test_df_labelled = test_df_labelled[["filenames_dest", "label", "polyp_id"]]
        
        
    else: 
        print("[INFO] Error in finding dataframe with data locations and labels")
        
        
    if config.UPSAMPLE: 
        print("[INFO] Performing dataset balancing ...")
        # Upsample minority classes 
        print("[INFO] Current label distribution:")
        print(train_df_labelled.label.value_counts())
        
        
        mean_count = round(np.mean(train_df_labelled.groupby('label').size())*config.UPSAMPLE_RATIO)
        print(f"[INFO] Target additional samples for each class { mean_count }")
        
        if class_type == 'paris': 
            #1) Separate majority and minority classes
            df_minority_LST = train_df_labelled[train_df_labelled.label=='LST'] 
            num_LST = len(df_minority_LST)
            print(num_LST)
            df_minority_peduncolate = train_df_labelled[train_df_labelled.label=='peduncolate']
            num_ped = len(df_minority_peduncolate)
            print(num_ped)
            df_majority = train_df_labelled[train_df_labelled['label'].isin(['sessile', 'flat'])]
            #2) Downsample majority class
            df_minority_LST_upsampled = resample(df_minority_LST, 
                                        replace=True,
                                        n_samples=num_LST + mean_count,     
                                        random_state=SEED)  
            df_minority_peduncolate_upsampled = resample(df_minority_peduncolate, 
                                                replace=True,
                                                n_samples=num_ped + mean_count,   
                                                random_state=SEED) 
            #3) Combine minority class with downsampled majority class
            train_df_labelled= pd.concat([df_minority_LST_upsampled, df_minority_peduncolate_upsampled, df_majority])
        elif class_type == 'nice': 
            
            df_minority_hyperplastic = train_df_labelled[train_df_labelled.label=='hyperplastic']
            num_hyper = len(df_minority_hyperplastic)
            df_minority_serrated = train_df_labelled[train_df_labelled.label=='serrated']
            num_serrated = len(df_minority_serrated)
            df_majority = train_df_labelled[train_df_labelled['label'].isin(['adenomatous'])]
            df_minority_hyperplastic_upsampled = resample(df_minority_hyperplastic, 
                                        replace=True,
                                        n_samples=num_hyper + mean_count,     
                                        random_state=SEED)  
            df_minority_serrated_upsampled = resample(df_minority_serrated, 
                                                replace=True,
                                                n_samples=num_serrated + mean_count,   
                                                random_state=SEED)
            
            #3) Combine minority class with downsampled majority class
            train_df_labelled= pd.concat([df_minority_hyperplastic_upsampled, df_minority_serrated_upsampled, df_majority])
        
        
        print("[INFO] Label Distribution after resampling:")
        print(train_df_labelled.label.value_counts())
        
        train_df_labelled.to_csv(dir_def_dataset+'/tmp_dataset_train_'+'.csv')
        test_df_labelled.to_csv(dir_def_dataset+'/tmp_dataset_test_'+'.csv')     
    return train_df_labelled, test_df_labelled

def from_dirs_to_frames(output_dir_dataset, df_sel_seqs, class_type, img_seq="image", smpl_rate= 1, encoded = False):
    """ Takes as input the directories of mp4 clipped videos and the list of selected sequences by the user and returns a dataset

    Args:
        output_dir_dataset (str): where to save dataset
        df_sel_seqs (dataframe): table containing selected sequences (depending on light choice and available data)
        class_type (str): indicates which label is considered: nice or paris
        img_seq (str, optional): indicates user choice of which dataset was chosen (sequence or image). Defaults to "image".
        smpl_rate (int, optional): sampling rate to extract images from videos (when sequence is selected as dataset). Defaults to 1.
        encoded (bool, optional): whether to encode the label into numbers or to use label name in string format. Defaults to False.

    Returns:
        output_df, data_reference_out_train, data_reference_out_test: output dataframe containing infromation on the saved data samples and directories
        
    """
    
    # for each row of the dataframe extract all the information and save accordingly
    filenames_dest = []
    label = []
    light = []
    polyp_id = []
    seq_id = []
    filename_source = []
    train_test_set = []
    for index, row in tqdm(df_sel_seqs.iterrows()):
        file_origin_dir = row['filenames_source']
        root, ext = os.path.splitext(file_origin_dir)
        seq_filename = root.split('/')
        # print(seq_filename)
        # print(root)
        # print(seq_filename[2])
        if encoded: 
            if class_type == 'paris': 
                current_label = paris_classes_inv[seq_filename[5]]
            elif class_type == 'nice': 
                current_label =  hist_type_inv[seq_filename[5]]
        else: 
            if class_type == 'paris': 
                current_label = seq_filename[5]
            elif class_type == 'nice': 
                current_label =  seq_filename[5]
                
                
        if img_seq == "sequence":
            if seq_filename[4] == 'train':
                file_dest_dir = output_dir_dataset+'/'+seq_filename[3]+'/'+'Frames' +'/' + seq_filename[4] + '/'+seq_filename[5]+'/'+ str(row['seq_names']) +'/' 
            elif seq_filename[4] == 'test':
                file_dest_dir = output_dir_dataset+'/'+seq_filename[3]+'/'+'Frames' +'/' + seq_filename[4] + '/'+ str(row['seq_names']) +'/' 
            else: 
                print("error in defining dirs!")
                
        elif img_seq == "image": 
            if seq_filename[4] == 'train':
                file_dest_dir = output_dir_dataset+'/'+seq_filename[3]+'/'+'Images'+'/' + seq_filename[4] + '/'
            elif seq_filename[4] == 'test': 
                file_dest_dir = output_dir_dataset+'/'+seq_filename[3]+'/'+'Images'+'/' + seq_filename[4] + '/'
            else: 
                # print(file_dest_dir)
                print("error in defining dirs!")
                
            smpl_rate = 1
        else: 
            print("Error, please define for what you want to use the dataset")
            break
        
        # print(file_dest_dir)
        if not os.path.isdir(file_dest_dir):
            os.makedirs(str(file_dest_dir))
        
        # Opens the Video file
        cap= cv2.VideoCapture(file_origin_dir+'.mp4')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
        # print(file_dest_dir)
        # print(total)
         
        ######################
        # TO DO: change so that images are saved only when current folder non empty
        # directory= os.listdir(file_dest_dir)
        # if len(directory) == 0: 
        #     print("Empty directory") 
        
        # else: 
        #     print("Not empty directory")
        ######################
        
        i=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            
            # the file name depends on the dataset we want to create
            if img_seq == "sequence":
                dest_name = 'img_{:06d}'.format(i) + '.png'
            else: 
                dest_name = seq_filename[5]+ '_'+ str(row['seq_names']) + '_{:06d}'.format(i) + '.png'
                
            
            dest_dir = os.path.join(file_dest_dir,dest_name)
            
            if img_seq == "image": 
                if i== 8 or i == total-3 or i==round(total/2): 
                    cv2.imwrite(dest_dir,frame)
                    filenames_dest.append(dest_dir)
                    label.append(current_label)
                    light.append(row['light'])
                    polyp_id.append(row['polyp_id'])
                    seq_id.append(row['seq_names'])
                    filename_source.append(file_origin_dir)
                    train_test_set.append(seq_filename[4])
                    # print(f'I am inside condition to save for {dest_dir}')

            else: 
                cv2.imwrite(dest_dir,frame)
                filenames_dest.append(dest_dir)
                label.append(current_label)
                light.append(row['light'])
                polyp_id.append(row['polyp_id'])
                seq_id.append(row['seq_names'])
                filename_source.append(file_origin_dir)
                train_test_set.append(seq_filename[4])

            i+=smpl_rate

        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.5)

            
    output_df = {'light': light, 'polyp_id': polyp_id, 'seq_id': seq_id, 'filename_source': filename_source, 'filenames_dest':filenames_dest, 'label':label, 'train_test_set': train_test_set}
    output_df = pd.DataFrame(output_df)
    # output_df = pd.concat([df_sel_seqs,adding_cols], axis=1)
    #df_sel_seqs.append(adding_cols, ignore_index = False)
    
    if img_seq == "image": 
        data_reference_out_train = output_dir_dataset+'/'+seq_filename[3]+'/'+'Images' +'/' + 'train'
        data_reference_out_test =  output_dir_dataset+'/'+seq_filename[3]+'/'+'Images' +'/' + 'test'
    elif img_seq == "sequence": 
        data_reference_out_train = output_dir_dataset+'/'+seq_filename[3]+'/'+'Frames' +'/' + 'train'
        data_reference_out_test =  output_dir_dataset+'/'+seq_filename[3]+'/'+'Frames' +'/' + 'test'
    else: 
        print("specify img_seq parameter or check spelling")
        
    return output_df, data_reference_out_train, data_reference_out_test

def selecting_dataset_files(dataframe_source, sequence_table, num_sequences = "all", light_chosen = "both"): 
    """Filtering the dataset using input choices.

    Args:
        dataframe_source (df): containing information on the data (label, directory ...)
        sequence_table (df): dataframe containing information on the sequences 
        num_sequences (str, optional): whether to take all or one sequences from each polyp. Defaults to "all".
        light_chosen (str, optional): indicates user choice of which light was selected. Defaults to "both".

    Returns:
        selected_seqs: returns only the seuqences that match input properties
    """
    
    # initialize some empty lists 
    polyp_ids = []
    lights = []
    file_dirs = []
    seq_names = []
    
    for i, row in tqdm(dataframe_source.iterrows(), total=dataframe_source.shape[0]): 

        seq_dir = row["dir_save"]
        # print(seq_dir)
        polyp_id = row['polyp_id']
        # print(f"this is the polyp id: {polyp_id}")
        sequence_id = row['seq_id']
        # print(f"this is the sequence id: {sequence_id}")
        # print(type(int(sequence_id)))
        seq_filename = str(str(polyp_id) + '_' + str(sequence_id))
        # print(sequence_table['LIGHT'])
        current_light = sequence_table['LIGHT'][int(sequence_id)]

        if light_chosen == current_light or light_chosen == "both": 
            # if the file was not removed its polyp id is save in a new list
            polyp_ids.append(polyp_id)
            lights.append(current_light)
            seq_names.append(seq_filename)
            file_dirs.append(seq_dir)
        elif light_chosen != current_light: 
            pass
        else: 
            print('Specify light_condition')
    
    # AT THIS POINT list_of_files is the correct output when num_sequences is set to all
    # otherwise a dataframe is created
    selected_seqs = pd.DataFrame({'light':lights, 'polyp_id': polyp_ids, 'seq_names': seq_names, 'filenames_source':file_dirs})
    
    if num_sequences == "all": 
        # print("the lenght of list of files is")
        # print(len(list_of_files))
        return selected_seqs
    
    elif num_sequences == "one": 
        
        # here only one sequence per light type and polyp is taken. 
        # selected_seqs = selecting_seqs.groupby(column).sample(n= selecting_seqs[column].value_counts().min(), random_state=SEED)
        if light_chosen == "W" or light_chosen == "NBI":
            selected_seqs.drop_duplicates(subset ="polyp_id", keep = False, inplace = True)
        else: 
            selected_seqs.drop_duplicates(subset = ['polyp_id','light'], keep = False, inplace = True)
        return selected_seqs
    
def labelled_dataset_creation(RSCS_table, polyp_table, hist_type_dict, paris_type_dict, sequence_table, classificartion_type, logs_dir, output_conv_mp4, output_clipped_mp4): 
    
    '''
    This function outputs a cropped version of the dataset which is referenced as video dataset. 
    
    Dependences: datetime, os, ffmpeg (used inside another function called here)
    NOTE: Install ffmpeg-python not ffmpeg
    
    Args: 
        RSCS_table: table of the database with information on intervention data
        polyp_table: table of the database with polyp data 
        hist_type_dict: dictionary with information on Histology encoding 
        paris_type_dict: dictionary with information on Paris encoding
        sequence_table: table with information on video segments
        classification_type: Nice or Paris classification
        logs_dir: directory where to save logs 
        output_folder_mp4: directory where to save cropped version of the dataset
        root_dir_mp4: where to save first version of cropped mp4 files
        
    Returns: dataset folder structure with mp4
    
    
    '''
    # The following piece of code outputs a cropped version of the dataset which has to be considered the starting reference. 
    # Following stages in the code depend on choices made in the current implementation
    missing = []
    directory_saved = []
    sequence_id = []
    polyp_id = []
    hist_type = []
    paris_type = []
    
    if not os.path.isdir(output_conv_mp4):
        os.makedirs(output_conv_mp4)
        print(f'Output conversion dir: {output_conv_mp4} directory created!')
                    
    for i, k in enumerate (tqdm(polyp_table['ID_RSCS'].values)) : 
        
        # STEP1: Look for intervention in which the polyp was found and extract directory
        current_row = RSCS_table.loc[str(k)]
        directory_RSCS = current_row['RSCS_DIRECTORY']
        # STEP2: take the polyp id and use it to retrieve all sequences and labels
        current_polyp_id = polyp_table.index.values[i]
        current_split = polyp_table.SET.values[i]
        # print(current_split)
        # return all the sequences corresponding to a specific polyp. 
        current_polyp_sequences = sequence_table[sequence_table['ID_POLYP']== str(current_polyp_id)]
        
        # STEP 3: identify directory based on train or test. Two separate folders 
        #print(current_split)
        if current_split == 'train': 
            output_dir_set = output_clipped_mp4 + '/' + str(classificartion_type) + '/train'
        else: 
            output_dir_set = output_clipped_mp4 + '/' + str(classificartion_type) + '/test'
        
        # print("current output dir set")
        # print(output_dir_set)
        # STEP 4: iterate over all sequences and retrive start, end and label. 
        # For each sequence a new clip is created and saved in a folder named after the label in the following format: 
        # root_dir_out / class_type / train (test) / classes
        if not current_polyp_sequences.empty:
            for j, id_seq in enumerate (current_polyp_sequences.index):

                # in order to correctly extract by index I have to extract the id_sequence of the first sequence for each polyp and then add j 
                # print(j)
                # print(f'this is sequence id {id_seq}')
                # 1. Take Time instant and window length and return start and end intervals for cropping

                tmp_time_instant = current_polyp_sequences['TIME_INSTANT_START'][id_seq]
                tmp_window_len = current_polyp_sequences['WINDOW_LEN'][id_seq]
                current_file_name = current_polyp_sequences['FILE_NAME'][id_seq]
                # 2. For each sequence use the label to define the output directory
                paris_class = polyp_table['TYPE'][current_polyp_id]
                # print(paris_class)
                histology = polyp_table['HISTOLOGY_TYPE'][current_polyp_id]
                # print(histology)
                
                input_dir_avi = str(directory_RSCS) + '/'+ str(current_file_name)+ '.avi'
                # print(input_dir_avi)
                output_dir_mp4 = output_conv_mp4+'/'+ str(current_file_name)
                

                
                # 3. Convert video into .mp4 (no resolution loss)
                if not os.path.isfile(output_dir_mp4+'.mp4'): 
                    convert_avi_to_mp4(input_dir_avi, output_dir_mp4)
                # else:
                    #print('The mp4 file was already generated in previous iterations')

                # 4. Clip the videos and save in output_dir 
                # here depending on the condition a directory is set. 
                # check the classification we want to perform and create two variables: 
                # output_file_dir : directory of the file
                # output_dir : directory of the folder used to make the directory
                
                if classificartion_type == 'paris': 
                    
                    output_file_dir = output_dir_set +'/'+ str(paris_type_dict[int(paris_class)]) + '/' + str(current_polyp_id) +'_' + str(id_seq)
                    output_dir = output_dir_set + '/'+str(paris_type_dict[int(paris_class)])
                elif classificartion_type == 'nice': 
                    output_file_dir = output_dir_set +'/'+ str(hist_type_dict[int(histology)]) + '/' + str(current_polyp_id) +'_' + str(id_seq)
                    output_dir = output_dir_set +'/'+str(hist_type_dict[int(histology)])
                else: 
                    print("specify classification type correctly !")
                    return 0


                    
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                    # print(f'Output dir: {output_dir} directory created!')
                
                directory_saved.append(output_file_dir)
                sequence_id.append(id_seq)
                polyp_id.append(current_polyp_id)
                hist_type.append(hist_type_dict[int(histology)])
                paris_type.append(paris_type_dict[int(paris_class)])
                
                
                if not os.path.isfile(output_file_dir+'.mp4'): 
                    time.sleep(1)
                    # print(output_dir)
                    # print(str(tmp_time_instant))
                    # print(str(tmp_window_len))
                    trim_video(output_dir_mp4, output_file_dir, str(tmp_time_instant), str(tmp_window_len))
                #else: 
                    #print(f'The target sequence {id_seq} for polyp {current_polyp_id} file was already generated in previous iterations')
        else: 
            #print(f'There are no polyp sequences for polyp {current_polyp_id}')
            missing.append(current_polyp_id)
            if len(missing) > 500: 
                print('There are too many missing sequences, double check csv and code!')
                error_code = 1
                break
    
    with open(logs_dir + '/missing_polyps.txt', 'w') as f:
        for i, v in enumerate(missing):
            f.write("The polyp with id {} has no sequences \n".format(v))
        
        if error_code == 1: 
            f.write("There was an issue with missing sequences: too many detected \n")
    
    
    dict_out = {'dir_save': directory_saved, 'seq_id': sequence_id, 'polyp_id': polyp_id, 'hist_t': hist_type, 'paris_t': paris_type }
    df_out =  pd.DataFrame(dict_out)
    return df_out


def convert_avi_to_mp4(avi_file_path, output_name):
    '''
    function to convert video from avi to mp4
    '''
    os.popen("ffmpeg -loglevel panic -i '{input}' -c:v copy -c:a copy -y  '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True

def trim_video(input_mp4, output_mp4, start, duration): 
    '''
    function to cut video based on start and duration of the clip
    '''
    os.popen("ffmpeg -loglevel panic -ss '{start}' -i '{input}.mp4' -to '{duration}' -c:v copy -c:a copy '{output}.mp4'".format(input = input_mp4, output = output_mp4, start = start, duration = duration))
    return True


def custom_train_test_split(RSCS_table, polyp_table, classification_type, logs_dir): 
    
    '''
    function to perform custom stratified train / test split. 
    Dependency: MultiLabelBinarizer, MultilabelStratifiedKFold, numpy
    
    Args: 
        RSCS_table: table of database with intervention data
        polyp_table: table of database with polyp data
    Returns: 
        train_df: polyp table training 
        test_df: polyp table test
    
    '''
    
    multi_label_list = []
    for i, row in RSCS_table.iterrows():
        patient_id = row['ID_PATIENT']
        id_RSCS = i 
        
        if classification_type == 'paris': 
            labels = list(polyp_table[polyp_table['ID_RSCS']==id_RSCS]["TYPE"])
        elif classification_type == 'nice': 
            labels = list(polyp_table[polyp_table['ID_RSCS']==id_RSCS]["HISTOLOGY_TYPE"])
        
        unique_labels = set(labels)
        unique_labels = list(unique_labels)
        multi_label_list.append(unique_labels)


    RSCS_table["multilabel"] = multi_label_list
    
    current_X = np.random.RandomState(seed=SEED).permutation(RSCS_table.index) #.tolist()
    classes_polyps = list(polyp_table['TYPE'].cat.categories)
    mlb = MultiLabelBinarizer(classes = classes_polyps) #classes=classes
    current_y = mlb.fit_transform(RSCS_table.multilabel)
    
    print('Train Validation Splitting ...')
    valid_kfold = MultilabelStratifiedKFold(n_splits=5, shuffle=False, random_state=None)

    fold = 0
    for train_indx, test_indx in valid_kfold.split(current_X, current_y): 
        fold = fold +1
        train_RSCS_list = []
        test_RSCS_list = []
        
        for i, tr_ix in enumerate(train_indx): 
            train_RSCS_list.append(current_X[tr_ix])
        
        for i, te_ix in enumerate(test_indx): 
            test_RSCS_list.append(current_X[te_ix])
        
        break
    
    print(f"The dataset is composed of {len(RSCS_table.index)}")
    print(f"The number of test samples is: {len(test_RSCS_list)}")
    print(f"The number of train samples is: {len(train_RSCS_list)}")
    
    set_col = []
    for i, row in polyp_table.iterrows(): 
        if row['ID_RSCS'] in (train_RSCS_list): 
            set_col.append('train')
        else: 
            set_col.append('test')

    polyp_table['SET'] = set_col      
    polyp_table['SET'] = polyp_table['SET'].astype("category")
    plyp_tbl_set = polyp_table 
    train_df = polyp_table[polyp_table['SET'] == 'train'].index
    test_df = polyp_table[polyp_table['SET'] == 'test'].index
    list_df = {'train':list(train_df), 'test':list(test_df)}
    with open(logs_dir+"/train_test_split.json", 'w') as outfile:
        outfile.write(json.dumps(list_df))
        
    return  plyp_tbl_set


def normalize_database(raw_dataframe, data_directory_GENIALCO, encoding_categorical_df, logs_dir, cols_map): 
    '''
    This code transforms the raw excel file with all the information on polyps into a normalized database structure. 
    This function should be rewritten if the source excel changes. 
    
    Dependences: pandas, datetime
    
    Args: 
        raw_dataframe: excel file generated by Operative Unit 
        data_directory_GENIALCO: directory where data is stored in folders sequentially
        logs_dir: output directory where to save information on the dataset
        cols_map: dictionary with the mapping of column names
    
    Returns: 
        patient_table
        RSCS_table 
        polyp_table 
        
    '''
    # this code only works for the current excel file structure. (25 ott 2021)
    # New code should be wirtten if columns change
    
    #############   RSCS TABLE   ############# 
    RSCS_list_cols = [*range(0,3), 7, 8, *range(15,28), *range(114,122)]
    intervention_data = raw_dataframe.iloc[:, RSCS_list_cols].copy()
    intervention_data.rename(columns={'ID': 'ID_RSCS'}, inplace=True)
    intervention_data = intervention_data.astype({"ID_RSCS": str})
    intervention_data['ID_RSCS'] = intervention_data['ID_RSCS'].str.replace(" ","")   
    # Add ID_PATIENT so to allow one-to-many relationships between patients and interventions
    id_patient = 'pt_' + intervention_data['ID_RSCS'].astype(str)
    intervention_data.insert(2, 'ID_PATIENT', id_patient) 
    # DECLARING DIRECTORY FOLDER
    dir_name_intervention = []
    for i, k in enumerate(intervention_data['ID_RSCS'].astype(str)): 
        dir_name_intervention.append(str(data_directory_GENIALCO+ str(i+1)))
        #print(dir_name_intervention)
    dir_name_intervention = np.resize(dir_name_intervention,len(dir_name_intervention))
    intervention_data.insert(1, 'RSCS_DIRECTORY', dir_name_intervention)
    
    # CHANGE COLUMN NAMES
    intervention_data.columns = intervention_data.columns.to_series().map(cols_map)
    # CONVERT DATA TYPES AND USE CATEGORICAL COLS
    # intervention_data = intervention_data.astype({'ID_PATIENT': str})
    intervention_data = intervention_data.convert_dtypes()
    
    # ENDOSCOPIST NAMES ARE ENCODED SO TO RESPECT PRIVACY
    intervention_data['ENDOSCOPIST'] = intervention_data['ENDOSCOPIST'].astype("category")
    print(intervention_data['ENDOSCOPIST'].cat.categories)
    intervention_data['ENDOSCOPIST'] = intervention_data['ENDOSCOPIST'].str.replace(" ","")
    
    intervention_data.update(intervention_data[list(encoding_categorical_df)].apply(lambda col: col.map(encoding_categorical_df[col.name])))
    intervention_data['ENDOSCOPIST'] = intervention_data['ENDOSCOPIST'].astype("category")
    
    # AI, PREPARATION, INTAKE, CECUM, BBPS_RIGHT, BBPS_TRANSV, 
    # BBPS_LT, ADVERSE_EVENT, POLYPS, ADENOMAS, SERRATED, CRC, FALSE_POSITIVES 
    # SHOULD ALL BE CATEGORICAL 
    intervention_data['AI'] = intervention_data['AI'].astype("category")
    intervention_data['PREPARATION'] = intervention_data['PREPARATION'].astype("category")
    intervention_data['INTAKE'] = intervention_data['INTAKE'].astype("category")
    intervention_data['CECUM'] = intervention_data['CECUM'].astype("category")
    intervention_data['BBPS_RIGHT'] = intervention_data['BBPS_RIGHT'].astype("category")
    intervention_data['BBPS_TRANSV'] = intervention_data['BBPS_TRANSV'].astype("category")
    intervention_data['BBPS_LT'] = intervention_data['BBPS_LT'].astype("category")
    intervention_data['ADVERSE_EVENT'] = intervention_data['ADVERSE_EVENT'].astype("category")
    intervention_data['POLYPS'] = intervention_data['POLYPS'].astype("category")
    intervention_data['ADENOMAS'] = intervention_data['ADENOMAS'].astype("category")
    intervention_data['SERRATED'] = intervention_data['SERRATED'].astype("category")
    intervention_data['CRC'] = intervention_data['CRC'].astype("category")
    intervention_data['FALSE_POSITIVES'] = intervention_data['FALSE_POSITIVES'].astype("category")
    
    # LOGS RSCS 
    RSCS_cols = intervention_data.columns
    with open(logs_dir + '/RSCS_cols.txt', 'w') as f:
        f.write(str(RSCS_cols))
    
    ############# PATIENT TABLE #############
    patient_list_cols = [*range(3,7), *range(9,15)] # unpacking ranges inside list for indexes that belong to intervention data
    patient_data = raw_dataframe.iloc[:, patient_list_cols].copy()
    patient_data.insert(1,'ID_PATIENT', intervention_data['ID_PATIENT'].values)
    patient_data = patient_data.set_index('ID_PATIENT')
    patient_data = patient_data.dropna(thresh=3, axis=0)
    patient_data_cols = patient_data.columns
        
    # NAMES OF THE COLUMNS ARE CHANGED
    patient_data.columns = patient_data.columns.to_series().map(cols_PATIENTS_new)
    # CATEGORICAL TYPES FOR SOME COLUMNS 
    patient_data['SEX'] = patient_data['SEX'].astype("category")
    patient_data['FIRST_RSCS'] = patient_data['FIRST_RSCS'].astype("category")
    patient_data['PAST_ADENOMAS'] = patient_data['PAST_ADENOMAS'].astype("category")
    patient_data['PAST_ADVANCED_ADENOMAS'] = patient_data['PAST_ADVANCED_ADENOMAS'].astype("category")
    # THE SYMPTOMS COLUMN MIGHT HAVE MORE THAN VALUE
    patient_data['SYMPTOMS'] = patient_data['SYMPTOMS'].astype("category")
    # DATE OF BIRTH SHOULD BECOME AGE (REFERENCE IS 1st MARCH 2020 START OF TRIAL)
    patient_data['DOB']= pd.to_datetime(patient_data['DOB'])
    def age(born):
        start_trial = datetime.date.fromisoformat('2020-03-01')
        return start_trial.year - born.year
    
    patient_data['AGE'] = patient_data['DOB'].apply(age)   
    # LOGS PATIENT
    with open(logs_dir + '/patient_data_cols.txt', 'w') as f:
        f.write(str(patient_data_cols))
        
    # DROP RSCS MISSING DATA
    # DONE AFTER SO TO PRESERVE RELATIONSHIPS IN PREVIOUS TABLE
    intervention_data = intervention_data.dropna(thresh=15, axis=0)
    
    ############# POLYP TABLE #############
    # Each intervention has all the polyps presented on the same row but different columns
    dictionary_polyps_tmp = {}

    for i, k in enumerate(intervention_data['ID_RSCS'].astype(str)): 
        # Step 1: if polyps are present read the number of polyps from the dataframe RSCS
        if str(intervention_data.iloc[i, 15]) == 'SI': 
            num_polipi = int(intervention_data.iloc[i, 16])
            
            # For each polyp save a dictionary
            for j in range(num_polipi):  
                polyp_data_tmp_dict = {}
                start = 29 + j*5
                end = 34 + j*5
                label_index_polyp = [*range(start,end)] 
                polyp_data = raw_dataframe.iloc[i, label_index_polyp].copy()
                polyp_data_tmp_dict['ID_RSCS'] = k
                polyp_data_tmp_dict['WHERE'] = polyp_data[0]
                polyp_data_tmp_dict['TYPE'] = polyp_data[1]
                polyp_data_tmp_dict['SIZE'] = polyp_data[2]
                polyp_data_tmp_dict['HISTOLOGY'] = polyp_data[3]
                polyp_data_tmp_dict['SEEN_BY'] = polyp_data[4]
                
                id_polyp = 'plyp_'+str(i+1)+'_'+ str(j+1)
                
                dictionary_polyps_tmp[id_polyp] = polyp_data_tmp_dict
                
    polyp_data = pd.DataFrame.from_dict(dictionary_polyps_tmp, orient='index')
    intervention_data = intervention_data.set_index('ID_RSCS')
    polyp_data = polyp_data.dropna(thresh=3, axis=0)

    # MODIFY TYPE OF COLUMNS 
    polyp_data['WHERE'] = polyp_data['WHERE'].astype("category")
    polyp_data['TYPE'] = polyp_data['TYPE'].astype("category")
    polyp_data['SIZE'] = polyp_data['SIZE'].astype("category")
    polyp_data['HISTOLOGY'] = polyp_data['HISTOLOGY'].astype("category")
    polyp_data['SEEN_BY'] = polyp_data['SEEN_BY'].astype("category")
    
    # MAP HISTOLOGIC TYPE 
    histologic_label = []
    for i, row in polyp_data.iterrows():
        
        adenomatous_list = [0,1,2,5, 9, '0.0', '1','2','5','9']
        hyperplastic_list = [6, 8, '6', '8', 'INFIAM']
        serrated_list = [3,4]
        if row["HISTOLOGY"] in adenomatous_list: 
            histologic_label.append(1)
        elif row["HISTOLOGY"] in hyperplastic_list:
            histologic_label.append(2)
        elif row["HISTOLOGY"] in serrated_list:
            histologic_label.append(3)
        else: 
            histologic_label.append(4)

    polyp_data["HISTOLOGY_TYPE"] = histologic_label
    polyp_data["HISTOLOGY_TYPE"] = polyp_data["HISTOLOGY_TYPE"].astype("category")

    # IF THERE ARE UNEXPECTED CATEGORIES THEY ARE REMOVED
    try: 
        category_error = polyp_data["TYPE"].cat.categories[4]
        indx_dr = polyp_data.loc[polyp_data["TYPE"] == category_error].index
        polyp_data = polyp_data.drop(indx_dr)
        polyp_data["TYPE"] = polyp_data.TYPE.cat.remove_unused_categories()
    
    except: 
        print('No errors in polyp categories')
        
    # LOGS POLYPS 
    polyp_data_cols = polyp_data.columns
    with open(logs_dir + '/polyp_data_cols.txt', 'w') as f:
        f.write(str(polyp_data_cols))
    
    rscs_table = intervention_data
    patient_table = patient_data
    polyp_table = polyp_data
        
    return rscs_table, patient_table, polyp_table


def sequence_table_from_csv(sequence_csv_directory, logs_dir): 
    
    '''
    Define sequence table from csv. 
    The csv is the result of the LABELLING PROCESS. 
    
    Dependences:  pandas
    
    Args: 
        sequence_csv_directory: dir where the csv is located
        logs_dir: output directory where to save logs
    
    Returns: 
        sequence_table
    '''
    # DEFINING SEQUENCE TABLE
    sequences_raw_data = pd.read_csv(sequence_csv_directory, sep=',',infer_datetime_format= True)
    seq_cols = list(sequences_raw_data.columns)
    sequences_raw_data = sequences_raw_data.dropna(thresh=5, axis=1)
    sequences_raw_data = sequences_raw_data.dropna(thresh=3, axis=0)

    # MODIFY TYPE OF COLUMNS
    sequences_raw_data['TIME_INSTANT_START'] = pd.to_datetime(sequences_raw_data['TIME_INSTANT_START'], format='%M:%S:%f')
    sequences_raw_data['TIME_INSTANT_START'] = pd.Series([val.time() for val in sequences_raw_data['TIME_INSTANT_START']])
    sequences_raw_data['WINDOW_LEN'] = pd.to_datetime(sequences_raw_data['WINDOW_LEN'],format='%M:%S:%f')
    sequences_raw_data['WINDOW_LEN'] = pd.Series([val.time() for val in sequences_raw_data['WINDOW_LEN']])
    sequences_raw_data['ID_SEQUENCE'] = sequences_raw_data.ID_SEQUENCE.astype(int)
    sequences_raw_data = sequences_raw_data.set_index('ID_SEQUENCE')
    sequences_raw_data['LIGHT'] = sequences_raw_data['LIGHT'].astype("category")
    sequences_raw_data_cols = sequences_raw_data.columns
    with open(logs_dir + '/sequences_raw_data_cols.txt', 'w') as f:
        f.write(str(sequences_raw_data_cols))
        
    sequence_table = sequences_raw_data
    
    return sequence_table

def save_tables(dir_def_dataset, logs_dir, output_df, light, img_or_seq, exp, rscs_tbl, patient_tbl, polyp_tbl, seqs_tbl): 
        
        # save an xlsx file for each of the tables of the dataset
        directories_pkl = []
        dir_save_json = logs_dir +'/categorical_encoding.json'
        directories_pkl.append(dir_save_json)
        seqs_tbl.to_excel(dir_def_dataset+'/sequence_table.xlsx')  
        rscs_tbl.to_excel(dir_def_dataset+'/RSCS_table.xlsx')
        patient_tbl.to_excel(dir_def_dataset+'/patient_table.xlsx')
        polyp_tbl.to_excel(dir_def_dataset+'/polyp_table.xlsx')
        output_df.to_csv(dir_def_dataset+'/dataset_'+img_or_seq+'_'+light+'_'+exp+'.csv')

        # save also in pkl format to retrive faster
        pkl_sequences_raw_data = dir_def_dataset+'/sequence_table.pkl'
        seqs_tbl.to_pickle(pkl_sequences_raw_data) 
        directories_pkl.append(pkl_sequences_raw_data)
        pkl_intervention_data = dir_def_dataset+'/RSCS_table.pkl'
        rscs_tbl.to_pickle(pkl_intervention_data)
        directories_pkl.append(pkl_intervention_data)
        pkl_patient_data = dir_def_dataset+'/patient_table.pkl'
        patient_tbl.to_pickle(pkl_patient_data)
        directories_pkl.append(pkl_patient_data)
        pkl_polyp_data = dir_def_dataset+'/polyp_table.pkl'
        polyp_tbl.to_pickle(pkl_polyp_data)
        directories_pkl.append(pkl_polyp_data)
        pkl_output_dataframe = dir_def_dataset+'/dataset_'+img_or_seq+'_'+light+'_'+exp+'.pkl'
        output_df.to_pickle(pkl_output_dataframe)
        directories_pkl.append(pkl_output_dataframe)

        with open(logs_dir + '/data_dirs_pkl.txt', 'w') as f:
            for i, v in enumerate(directories_pkl):
                f.write("{}\n".format(v))

        output_df["label_one_hot"] = output_df["label"].str.get_dummies().values.tolist()
        train_df_labelled = output_df[output_df["train_test_set"]=='train']
        train_df_labelled = train_df_labelled[["filenames_dest", "label", "polyp_id", "label_one_hot"]]

        test_df_labelled = output_df[output_df["train_test_set"]=='test']
        test_df_labelled = test_df_labelled[["filenames_dest", "label","polyp_id", "label_one_hot"]]
        train_df_labelled.to_csv(dir_def_dataset+'/dataset_train_'+ str(exp)+'.csv')
        test_df_labelled.to_csv(dir_def_dataset+'/dataset_test_'+ str(exp)+'.csv')
        
        return True
