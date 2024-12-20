import numpy as np
import os
import datetime
import json
import pickle
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def preprocess_dataset(dataset_name, train_length, test_length, skip_length, history_length):
    """
    Preprocesses and saves a specified dataset for forecasting task, including training, testing, and validation sets.
    This function saves the corresponding processed splits (train and test), and the mean and standard deviation for the training set used for normalization.

    
    Parameters:
    - dataset_name (str): The name of the dataset. It should be one of the following: electricity, solar, taxi, traffic or wiki.
    - train_length (int): The length of the training sequences. It refers to the total number of timestamps in  training and validation sets.
    - test_length (int): The length of the testing sequences. It refers to the total number of timestamps in the test set.
    - skip_length (int): The number of sequences to skip between training and testing. The number of timestamps to skip in the test set (we are evaluating only on a subset).
    - history_length (int): The length of the historical data to consider. The total number of timestamps to use as history window in every MTS.


    """

    path_train = f'./data/{dataset_name}/{dataset_name}_nips/train/data.json' #train
    path_test = f'./data/{dataset_name}/{dataset_name}_nips/test/data.json' #test
    

    main_data=[]
    mask_data=[]
    
    hour_data=None
    

    with open(path_train, 'r') as file:
        data_train = [json.loads(line) for line in file]

    with open(path_test, 'r') as file:
        data_test = [json.loads(line) for line in file]

    ## Prepare Train Sequences
    for obj in data_train:
        tmp_data = np.array(obj['target'])
        tmp_mask = np.ones_like(tmp_data)

        if len(tmp_data) == train_length and hour_data is None:
            c_time = datetime.datetime.strptime(obj['start'],'%Y-%m-%d %H:%M:%S')
            hour_data = []
            day_data = []
            time_data = []
            for k in range(train_length):
                time_data.append(c_time)
                hour_data.append(c_time.hour)
                day_data.append(c_time.weekday())
                c_time = c_time + datetime.timedelta(hours=1)
        else:
            if len(tmp_data) != train_length: #fill NA by 0 
                tmp_padding = np.zeros(train_length-len(tmp_data))
                tmp_data = np.concatenate([tmp_padding,tmp_data])
                tmp_mask = np.concatenate([tmp_padding,tmp_mask])

        
        main_data.append(tmp_data)
        mask_data.append(tmp_mask)

    ## Prepare Test Sequences
    if dataset_name != "solar":
        cnt = 0
        ind = 0

        for line in data_test:
            cnt+=1
            if cnt <=skip_length:
                continue
            tmp_data = np.array(line['target'])
            tmp_data = tmp_data[-test_length-history_length:] 

            tmp_mask = np.ones_like(tmp_data)

            main_data[ind] = np.concatenate([main_data[ind],tmp_data])
            mask_data[ind] = np.concatenate([mask_data[ind],tmp_mask])
            c_time = datetime.datetime.strptime(obj['start'],'%Y-%m-%d %H:%M:%S')
            for i in range(test_length+history_length):
                time_data.append(c_time)
                hour_data.append(c_time.hour)
                day_data.append(c_time.weekday())
                c_time = c_time + datetime.timedelta(hours=1)
            ind += 1
    main_data = np.stack(main_data,-1)
    mask_data = np.stack(mask_data,-1)
    print('Main data shape', main_data.shape)
    ## Save means
    mean_data = main_data[:-test_length-history_length].mean(0)
    std_data = main_data[:-test_length-history_length].std(0)


    ## Save means
    paths=f'./data/{dataset_name}/{dataset_name}_nips/meanstd.pkl' 
    if os.path.isfile(paths) == False:
        with open(paths, 'wb') as f:
            pickle.dump([mean_data,std_data],f)
            
    ## Save sequences
    paths=f'./data/{dataset_name}/{dataset_name}_nips/data.pkl' 
    if os.path.isfile(paths) == False:
        with open(paths, 'wb') as f:
            pickle.dump([main_data,mask_data],f)


def preprocess_taxi(train_length, test_length, skip_length, history_length):
    """
    Specialized preprocessing function for the taxi dataset, similar to preprocess_dataset but tailored to its structure. 
    
    Parameters:
    - train_length (int): The length of the training sequences. It refers to the total number of timestamps in  training and validation sets.
    - test_length (int): The length of the testing sequences. It refers to the total number of timestamps in the test set.
    - skip_length (int): The number of sequences to skip between training and testing. The number of timestamps to skip in the test set (we are evaluating only on a subset).
    - history_length (int): The length of the historical data to consider. The total number of timestamps to use as history window in every MTS.
    """

    path_train = f'./data/taxi/taxi_nips/train/data.json' #train
    path_test = f'./data/taxi/taxi_nips/test/data.json' #test
    
    
    main_data=[]
    mask_data=[]
    
    hour_data=None
    

    with open(path_train, 'r') as file:
        data_train = [json.loads(line) for line in file]

    with open(path_test, 'r') as file:
        data_test = [json.loads(line) for line in file]
    
    ## Prepare Train Sequences
    for obj in data_train:
        tmp_data = np.array(obj['target'])
        tmp_mask = np.ones_like(tmp_data)

        if len(tmp_data) == train_length and hour_data is None:
            c_time = datetime.datetime.strptime(obj['start'],'%Y-%m-%d %H:%M:%S')
            hour_data = []
            day_data = []
            time_data = []
            for k in range(train_length):
                time_data.append(c_time)
                hour_data.append(int(c_time.hour+c_time.minute/30))
                day_data.append(c_time.weekday())
                c_time = c_time + datetime.timedelta(minutes=30)
        else:
            if len(tmp_data) != train_length: #fill NA by 0 
                tmp_padding = np.zeros(train_length-len(tmp_data))
                tmp_data = np.concatenate([tmp_padding,tmp_data])
                tmp_mask = np.concatenate([tmp_padding,tmp_mask])

        
        main_data.append(tmp_data)
        mask_data.append(tmp_mask)

    ## Prepare Test Sequences
    cnt = 0
    ind = 0

    for line in data_test:
        cnt+=1
        if cnt <=skip_length:
            continue
        tmp_data = np.array(line['target'])
        tmp_data = tmp_data[-test_length-history_length:] 
        
        tmp_mask = np.ones_like(tmp_data)
        
        main_data[ind] = np.concatenate([main_data[ind],tmp_data])
        mask_data[ind] = np.concatenate([mask_data[ind],tmp_mask])
        c_time = datetime.datetime.strptime(obj['start'],'%Y-%m-%d %H:%M:%S')
        for i in range(test_length+history_length):
            time_data.append(c_time)
            hour_data.append(c_time.hour)
            day_data.append(c_time.weekday())
            c_time = c_time + datetime.timedelta(minutes=30)
        ind += 1
    
    main_data = np.stack(main_data,-1)
    mask_data = np.stack(mask_data,-1)
    

    ## Save mean
    mean_data = main_data[:-test_length-history_length].mean(0)
    std_data = main_data[:-test_length-history_length].std(0)

    ## Save means
    paths=f'./data/taxi/taxi_nips/meanstd.pkl' 
    if os.path.isfile(paths) == False:
        with open(paths, 'wb') as f:
            pickle.dump([mean_data,std_data],f)

    ## Save sequences
    paths=f'./data/taxi/taxi_nips/data.pkl' 
    if os.path.isfile(paths) == False:
        with open(paths, 'wb') as f:
            pickle.dump([main_data,mask_data],f)
    
def preprocess_wiki(train_length, test_length, skip_length, history_length):
    """
    Specialized preprocessing function for the wiki dataset, similar to preprocess_dataset but tailored to its structure.
    
    Parameters:
    - train_length (int): The length of the training sequences. It refers to the total number of timestamps in  training and validation sets.
    - test_length (int): The length of the testing sequences. It refers to the total number of timestamps in the test set.
    - skip_length (int): The number of sequences to skip between training and testing. The number of timestamps to skip in the test set (we are evaluating only on a subset).
    - history_length (int): The length of the historical data to consider. The total number of timestamps to use as history window in every MTS.
    """
    path_train = f'./data/wiki/wiki_nips/train/data.json' #train
    path_test = f'./data/wiki/wiki_nips/test/data.json' #test
    
    
    main_data=[]
    mask_data=[]
    
    hour_data=None
    

    with open(path_train, 'r') as file:
        data_train = [json.loads(line) for line in file]

    with open(path_test, 'r') as file:
        data_test = [json.loads(line) for line in file]

    ## Prepare Train Sequences
    for obj in data_train:
        tmp_data = np.array(obj['target'])
        tmp_mask = np.ones_like(tmp_data)

        if len(tmp_data) == train_length and hour_data is None:
            c_time = datetime.datetime.strptime(obj['start'],'%Y-%m-%d %H:%M:%S')
            hour_data = []
            day_data = []
            time_data = []
            for k in range(train_length):
                time_data.append(c_time)
                hour_data.append(c_time.hour)
                day_data.append(c_time.weekday())
                c_time = c_time + datetime.timedelta(days=1)
        else:
            if len(tmp_data) != train_length: #fill NA by 0 
                tmp_padding = np.zeros(train_length-len(tmp_data))
                tmp_data = np.concatenate([tmp_padding,tmp_data])
                tmp_mask = np.concatenate([tmp_padding,tmp_mask])

        
        main_data.append(tmp_data)
        mask_data.append(tmp_mask)

    ## Prepare Test Sequences
    cnt = 0
    ind = 0

    for line in data_test:
        cnt+=1
        if cnt <=skip_length:
            continue
        tmp_data = np.array(line['target'])
        tmp_data = tmp_data[-test_length-history_length:] 
        
        tmp_mask = np.ones_like(tmp_data)
        
        main_data[ind] = np.concatenate([main_data[ind],tmp_data])
        mask_data[ind] = np.concatenate([mask_data[ind],tmp_mask])
        c_time = datetime.datetime.strptime(obj['start'],'%Y-%m-%d %H:%M:%S')
        
        ind += 1
    main_data = np.stack(main_data[-2000:],-1)
    mask_data = np.stack(mask_data[-2000:],-1)
    
    mean_data = main_data[:-test_length-history_length].mean(0)
    std_data = main_data[:-test_length-history_length].std(0)

    ## Save means
    paths=f'./data/wiki/wiki_nips/meanstd.pkl' 
    if os.path.isfile(paths) == False:
        with open(paths, 'wb') as f:
            pickle.dump([mean_data,std_data],f)

    ## Save sequences
    paths=f'./data/wiki/wiki_nips/data.pkl' 
    if os.path.isfile(paths) == False:
        with open(paths, 'wb') as f:
            pickle.dump([main_data,mask_data],f)


class MyForecasting_Dataset(Dataset):
    def __init__(self, dataset_name, train_length, skip_length, valid_length, test_length, pred_length, history_length, is_train):
        self.train_length = train_length
        self.history_length = history_length
        self.pred_length = pred_length

        self.val_pct = valid_length / 100.
        self.test_pct = test_length / 100.
        self.data_type = dataset_name
        self.seq_length = self.history_length + self.pred_length

        data_path = f'./data/{self.data_type}/{self.data_type}/data.pkl'
        self.main_data_raw = pd.read_pickle(data_path)
        self.mask_data = 1 - self.main_data_raw.isna().astype(int).values
        self.main_data = self.main_data_raw.fillna(0)

        self.tvt_indices = self.get_train_test_from_df()

        if is_train == 0: # test
            self.use_index = self.tvt_indices['test']
            self.mode = 'test'
        elif is_train == 1: # train
            self.use_index = self.tvt_indices['train']
            self.mode = 'train'
        elif is_train == 2: # val
            self.use_index = self.tvt_indices['val']
            self.mode = 'val'
        elif is_train == 3: # predict
            self.use_index = self.tvt_indices['predict']
            self.mode = 'predict'

        self.mean_data = self.main_data.iloc[self.tvt_indices['train']].mean().values
        self.std_data = self.main_data.iloc[self.tvt_indices['train']].std().values

        # standardize the data
        self.main_data = ((self.main_data - self.mean_data) / np.maximum(1e-5, self.std_data)).values

    def get_train_test_from_df(self):

        train_indices = []
        val_indices = []
        test_indices = []
        predict_indices = []

        i = 0

        for geo, chunk in self.main_data.groupby('geo'):
            '''
                generate the train and test indices (aka sequence start index) of a given data frame.
                we group by geo, and within that, split into train and test sections
                according to the provided input_steps, forecast_steps, and val/test_pct
            '''
            chunk_length = len(chunk)
            temp = np.arange(chunk_length)
            n_val_steps = int(chunk_length * self.val_pct)
            n_test_steps = int(chunk_length * self.test_pct) 
            
            # adjust the number of seq_length blocks we need to handle
            if n_val_steps == 0:
                n_seq_length = 1
            else:
                n_seq_length = 2


            # split into "train" and "test/val" sections. anything before the test/val block is considered train
            test_val_len = n_test_steps + n_val_steps + (self.seq_length * n_seq_length)
            train_section = temp[:-test_val_len]
            test_val_section = temp[-test_val_len:]

            # within the test/val block, split into test and val sections according to test length (val length may be zero)
            test_len = n_test_steps + self.seq_length
            val_section = test_val_section[:-test_len]
            test_section = test_val_section[-test_len:]

            # for the predict section, we want to go all the way to the end of the data
            predict_len = n_test_steps + self.history_length
            predict_section = test_val_section[-predict_len:]

            # add the index of the start of each sequence to the appropriate list
            train_indices.extend([i + idx for idx in train_section[:-self.seq_length]])
            val_indices.extend([i + idx for idx in val_section[:-self.seq_length]])
            test_indices.extend([i + idx for idx in test_section[:-self.seq_length]])
            predict_indices.extend([i + idx for idx in predict_section])
            
            i += chunk_length
        return {
            'train': train_indices, 
            'val': val_indices, 
            'test': test_indices,
            'predict': predict_indices
        }
    

    def __getitem__(self, orgindex):
        """
        Gets the MTS at the specified index.
        
        Parameters:
        - orgindex (int): The index of the MTS (index of the start timestamp of the sequence).
        
        Returns:
        - dict: A dictionary containing 'observed_data', 'observed_mask', 'gt_mask', 'timepoints', and 'feature_id'.
        """
        index = self.use_index[orgindex]
        target_mask = self.mask_data[index:index+self.seq_length].copy()
        target_mask[-self.pred_length:] = 0. 
        s = {
            'observed_data': self.main_data[index:index+self.seq_length],
            'observed_mask': self.mask_data[index:index+self.seq_length],
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_length) * 1.0, 
            'feature_id': np.arange(self.main_data.shape[1]) * 1.0, 
        }

        return s
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
        - int: The total number of samples.
        """
        return len(self.use_index)       

# class Forecasting_Dataset(Dataset):
#     """
#     A PyTorch Dataset class for loading and preparing forecasting data.
    
#     Parameters:
#     - dataset_name (str): The name of the dataset. One of the following: electricity, solar, traffic, taxi or wiki.
#     - train_length, skip_length, valid_length, test_length, pred_length, history_length (int): Parameters defining the dataset structure and lengths of different segments as described in the processing functions.
#     - is_train (int): Indicator of the dataset split (0 for test, 1 for train, 2 for valid).
#     """
#     def __init__(self,  dataset_name, train_length, skip_length, valid_length, test_length, pred_length, history_length, is_train):
#         self.history_length = history_length
#         self.pred_length = pred_length
#         self.test_length = test_length
#         self.valid_length = valid_length
#         self.data_type = dataset_name
#         self.seq_length = self.pred_length+self.history_length

        
#         paths = f'./data/{self.data_type}/{self.data_type}/data.pkl' 
#         mean_path = f'./data/{self.data_type}/{self.data_type}/meanstd.pkl'

#         if dataset_name == 'euro_AT':
#             with open(paths, 'rb') as f:
#                 self.main_data,self.mask_data=pickle.load(f)
#             with open(mean_path, 'rb') as f:
#                 self.mean_data,self.std_data=pickle.load(f)        
#         else: 
#             if dataset_name == 'taxi':
#                 preprocess_taxi(train_length, test_length, skip_length, history_length)
#             elif dataset_name == 'wiki':
#                 preprocess_wiki(train_length, test_length, skip_length, history_length)
#             else:
#                 preprocess_dataset(dataset_name, train_length, test_length, skip_length, history_length)

#             paths = f'./data/{self.data_type}/{self.data_type}_nips/data.pkl' 
#             mean_path = f'./data/{self.data_type}/{self.data_type}_nips/meanstd.pkl'
#             with open(paths, 'rb') as f:
#                 self.main_data,self.mask_data=pickle.load(f)
#             with open(mean_path, 'rb') as f:
#                 self.mean_data,self.std_data=pickle.load(f)
            
#         self.main_data = (self.main_data - self.mean_data) / np.maximum(1e-5,self.std_data)

#         data_length = len(self.main_data)
#         if is_train == 0: #test
#             start = data_length - self.seq_length - self.test_length + self.pred_length
#             end = data_length - self.seq_length + self.pred_length
#             self.use_index = np.arange(start,end,self.pred_length)
#             print('Test', start, end)
            
#         if is_train == 2: #valid 
#             start = data_length - self.seq_length - self.valid_length - self.test_length + self.pred_length
#             end = data_length - self.seq_length - self.test_length + self.pred_length
#             self.use_index = np.arange(start,end,self.pred_length)
#             print('Val', start, end)
#         if is_train == 1:
#             start = 0
#             end = data_length - self.seq_length - self.valid_length - self.test_length + 1
#             self.use_index = np.arange(start,end,1)
#             print('Train', start, end)
        
        
#     def __getitem__(self, orgindex):
#         """
#         Gets the MTS at the specified index.
        
#         Parameters:
#         - orgindex (int): The index of the MTS (index of the start timestamp of the sequence).
        
#         Returns:
#         - dict: A dictionary containing 'observed_data', 'observed_mask', 'gt_mask', 'timepoints', and 'feature_id'.
#         """
#         index = self.use_index[orgindex]
#         target_mask = self.mask_data[index:index+self.seq_length].copy()
#         target_mask[-self.pred_length:] = 0. 
#         s = {
#             'observed_data': self.main_data[index:index+self.seq_length],
#             'observed_mask': self.mask_data[index:index+self.seq_length],
#             'gt_mask': target_mask,
#             'timepoints': np.arange(self.seq_length) * 1.0, 
#             'feature_id': np.arange(self.main_data.shape[1]) * 1.0, 
#         }

#         return s
    
#     def __len__(self):
#         """
#         Returns the total number of samples in the dataset.
        
#         Returns:
#         - int: The total number of samples.
#         """
#         return len(self.use_index)
    
class MyForecasting_Dataset(Dataset):
    def __init__(self, dataset_name, train_length, skip_length, valid_length, test_length, pred_length, history_length, is_train, prediction_run_id=None):
        self.train_length = train_length
        self.history_length = history_length
        self.pred_length = pred_length

        self.val_pct = valid_length / 100.
        self.test_pct = test_length / 100.
        self.data_type = dataset_name
        self.seq_length = self.history_length + self.pred_length

        if prediction_run_id is not None:
            data_path = f'./save/Forecasting/euro_all_countries/n_samples_100_run_{prediction_run_id}_linear_False_sample_feat_True/predictions/raw_df.pkl'
        else:
            data_path = f'./data/{self.data_type}/{self.data_type}/data.pkl'
        self.main_data_raw = pd.read_pickle(data_path)
        self.mask_data = 1 - self.main_data_raw.isna().astype(int).values
        self.main_data = self.main_data_raw.fillna(0)

        self.tvt_indices = self.get_train_test_from_df()

        if is_train == 0: # test
            self.use_index = self.tvt_indices['test']
            self.mode = 'test'
        elif is_train == 1: # train
            self.use_index = self.tvt_indices['train']
            self.mode = 'train'
        elif is_train == 2: # val
            self.use_index = self.tvt_indices['val']
            self.mode = 'val'
        elif is_train == 3: # predict
            self.use_index = self.tvt_indices['predict']
            self.mode = 'predict'

        # create a set of all indices betweeen start and end for each element of self.tvt_indices
        self.all_train_indices = list(set(np.concatenate([np.arange(start, end) for start, end in self.tvt_indices['train']])))

        # NOTE: means and std are calculated on the first value of every series, not the entire series
        # a quick analsysis showed that they're off by no more than ~10% for any feature
        # to fix, use `self.all_train_indices`
        self.mean_data = self.main_data.iloc[self.tvt_indices['train'][:,0]].mean().values
        self.std_data = self.main_data.iloc[self.tvt_indices['train'][:,0]].std().values

        # standardize the data
        self.main_data = ((self.main_data - self.mean_data) / np.maximum(1e-5, self.std_data)).values


        # self.mean_data = self.main_data.iloc[self.tvt_indices['train'][:,0]].mean().values
        # self.std_data = self.main_data.iloc[self.tvt_indices['train'][:,0]].std().values

        # # standardize the data
        # self.main_data = ((self.main_data - self.mean_data) / np.maximum(1e-5, self.std_data)).values

    def get_train_test_from_df(self):

        train_indices = []
        val_indices = []
        test_indices = []
        predict_indices = []

        i = 0

        for geo, chunk in self.main_data.groupby('geo'):
            '''
                generate the train and test indices (aka sequence start index) of a given data frame.
                we group by geo, and within that, split into train and test sections
                according to the provided input_steps, forecast_steps, and val/test_pct
            '''
            chunk_length = len(chunk)

            temp = np.arange(chunk_length)
            n_val_steps = int(chunk_length * self.val_pct)
            n_test_steps = int(chunk_length * self.test_pct) 
            
            # adjust the number of seq_length blocks we need to handle
            if n_val_steps == 0:
                n_seq_length = 1
            else:
                n_seq_length = 2


            # split into "train" and "test/val" sections. anything before the test/val block is considered train
            test_val_len = n_test_steps + n_val_steps + (self.seq_length * n_seq_length)
            train_section = temp[:-test_val_len]
            test_val_section = temp[-test_val_len:]

            # within the test/val block, split into test and val sections according to test length (val length may be zero)
            test_len = n_test_steps + self.seq_length
            val_section = test_val_section[:-test_len]
            test_section = test_val_section[-test_len:]

            # add the start and end indices of each sequence to the respective lists
            train_indices.extend([[i + idx, (i + idx + self.seq_length)] for idx in train_section[:-self.seq_length]])
            val_indices.extend([[i + idx, (i + idx + self.seq_length)] for idx in val_section[:-self.seq_length]])
            test_indices.extend([[i + idx, (i + idx + self.seq_length)] for idx in test_section[:-self.seq_length]])

            # for prediction, we go right up to the end of the sequence, and use min to avoid going over into the next chunk
            predict_indices.extend([[i + idx, (i + min(chunk_length, idx + self.seq_length))] for idx in test_section[:-self.history_length]])
            
            # increment the index by the length of the chunk
            i += chunk_length

        return {
            'train': np.array(train_indices), 
            'val': np.array(val_indices), 
            'test': np.array(test_indices),
            'predict': np.array(predict_indices)
        }
    
    def pad_sample(self, sample):
        """
        Pads a given matrix of shape (n, m) to shape (self.seq_length, m) if n < self.seq_length.
        
        Parameters:
        - sample (np.array): The sample matrix to be padded.
        
        Returns:
        - np.array: The padded matrix.
        """
        if sample.shape[0] < self.seq_length:
            pad = np.zeros((self.seq_length - sample.shape[0], sample.shape[1]))
            sample = np.concatenate([sample, pad], axis=0)
        return sample

    def __getitem__(self, orgindex):
        """
        Gets the MTS at the specified index and pads it if necessary.
        
        Parameters:
        - orgindex (int): The index of the MTS (index of the start timestamp of the sequence).
        
        Returns:
        - dict: A dictionary containing 'observed_data', 'observed_mask', 'gt_mask', 'timepoints', and 'feature_id'.
        """
        [start, end] = self.use_index[orgindex]
        
        # Ensure that the sequences are padded to seq_length if necessary
        observed_data = self.main_data[start:end]
        observed_mask = self.mask_data[start:end]
        
        # Create the target mask and set the last pred_length time steps to 0 (ground truth mask)
        target_mask = self.mask_data[start:end].copy()
        target_mask[-self.pred_length:] = 0.
        
        # Pad the data, mask, and ground truth mask if the sequence is shorter than seq_length
        if (end - start) < self.seq_length:
            observed_data = self.pad_sample(observed_data)
            observed_mask = self.pad_sample(observed_mask)
            target_mask = self.pad_sample(target_mask)
        
        # Create the sample dictionary
        s = {
            'observed_data': observed_data,
            'observed_mask': observed_mask,
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_length) * 1.0,  # Assuming timepoints are evenly spaced
            'feature_id': np.arange(self.main_data.shape[1]) * 1.0,  # Feature IDs are the columns
        }
        
        return s
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
        - int: The total number of samples.
        """
        return len(self.use_index)       


def get_dataloader_forecasting(dataset_name, train_length, skip_length, valid_length=24*5, test_length =24*7, pred_length=24, history_length=168, batch_size=8, device='cuda:0', prediction_run_id=None):
        """
        Prepares DataLoader objects for the forecasting datasets.
        
        Parameters:
        - dataset_name (str): The name of the dataset.
        - train_length, skip_length, valid_length, test_length, pred_length, history_length, batch_size (int): Various parameters defining dataset and DataLoader configurations.
        - device (str): The device to use for loading tensors.
        
        Returns:
        - Tuple[DataLoader, DataLoader, DataLoader, Tensor, Tensor]: Training, validation, and testing DataLoaders, along with scale and mean scale tensors used for normalization.
        """

        train_dataset = MyForecasting_Dataset(dataset_name, train_length, skip_length, valid_length, test_length, pred_length, history_length, is_train=1, prediction_run_id=prediction_run_id)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        valid_dataset = MyForecasting_Dataset(dataset_name, train_length, skip_length, valid_length, test_length, pred_length, history_length, is_train=2, prediction_run_id=prediction_run_id)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        test_dataset = MyForecasting_Dataset(dataset_name, train_length, skip_length, valid_length, test_length, pred_length, history_length, is_train=0, prediction_run_id=prediction_run_id)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # predict_dataset = MyForecasting_Dataset(dataset_name, train_length, skip_length, valid_length, test_length, pred_length, history_length, is_train=3)
        predict_dataset = MyForecasting_Dataset(dataset_name, train_length, skip_length, valid_length, test_length, pred_length, history_length, is_train=3, prediction_run_id=prediction_run_id)
        predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)


        # scaler = torch.from_numpy(train_dataset.std_data).to(device).float()
        scaler = torch.from_numpy(train_dataset.std_data.astype(np.float32)).to(device)
        mean_scaler = torch.from_numpy(train_dataset.mean_data.astype(np.float32)).to(device)
        all_indices = {
#            split: list(train_dataset.main_data_raw.index[idx]) for split, idx in train_dataset.tvt_indices.items()
            split: list(train_dataset.main_data_raw.index[idx[:,0]]) for split, idx in train_dataset.tvt_indices.items()
        }
        df_indices = pd.DataFrame([
            [split, idx[0], idx[1]] for split, indices in all_indices.items()
            for idx in indices
            ], columns=['split', 'country', 'time_period']
        )
        raw_df = train_dataset.main_data_raw
        return train_loader, valid_loader, test_loader, scaler, mean_scaler, df_indices, raw_df, predict_loader