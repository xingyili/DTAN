import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import OneHotEncoder

class FluDataset(Dataset):
    '''
    edge_index: (2, n) n is the number of edges in the graph
    '''
    def __init__(self, x, y, at_name=None, sr_name=None):
        self.x = x
        self.y = y
        self.at_name = at_name
        self.sr_name = sr_name
        self.at_ohe = None
        self.sr_ohe = None
        self.at_name_encoded = None
        self.sr_name_encoded = None
        
        # self.use_sparse = use_sparse

    def __len__(self): 
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        if self.y is not None:
            return torch.tensor(self.x[idx]).float(), torch.tensor(self.y[idx]), torch.tensor(self.at_name_encoded[idx]).float(), torch.tensor(self.sr_name_encoded[idx]).float()
        else:
            return torch.tensor(self.x[idx]).float(), torch.tensor(-1).float(), torch.tensor(self.at_name_encoded[idx].toarray().reshape(-1)).float(), torch.tensor(self.sr_name_encoded[idx].toarray().reshape(-1)).float()
    def fit_ohe(self):
        self.at_ohe = OneHotEncoder(handle_unknown='ignore')
        self.sr_ohe = OneHotEncoder(handle_unknown='ignore')
        self.at_name_encoded = self.at_ohe.fit_transform( [[x] for x in self.at_name]).toarray()
        self.sr_name_encoded = self.sr_ohe.fit_transform( [[x] for x in self.sr_name]).toarray()

    def transform_ohe(self):
        self.at_name_encoded = self.at_ohe.transform( [[x] for x in self.at_name]).toarray()
        self.sr_name_encoded = self.sr_ohe.transform( [[x] for x in self.sr_name]).toarray()

    def subset(self, sub_idx):
        sub_at_name = None
        sub_sr_name = None
        if self.at_name is not None and self.sr_name is not None:
            sub_at_name = [self.at_name[i] for i in sub_idx]
            sub_sr_name = [self.sr_name[i] for i in sub_idx]
        return FluDataset(self.x[sub_idx], self.y[sub_idx], sub_at_name, sub_sr_name)

def contains_invalid_chars(s, valid_chars):
    return bool(set(s) - valid_chars)

def check_gap_percentage(s, threshold=3e-2):
    gap_count = s.count('-')
    total_length = len(s)
    dash_percentage = gap_count / total_length
    return dash_percentage >= threshold

def filter_data(seq_df, anti_df, args):
    # 这里筛选掉带有'-', 'B', 'J', 'Z', 'X'的序列，并更新毒株索引
    del_idx = []
    valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
    for i in range(len(seq_df)):
        if contains_invalid_chars(seq_df.iloc[i, 2], valid_chars) or check_gap_percentage(seq_df.iloc[i, 2]):
            del_idx.append(i)
    seq_df = seq_df.drop(del_idx)
    seq_df = seq_df.reset_index(drop=True)
    seq_df['new_id'] = seq_df.index # 获取新的病毒索引
    seq_df = seq_df.set_index("id")
    idx_dict = seq_df['new_id'].to_dict()
    anti_df['at_index'] = anti_df['at_index'].map(idx_dict)
    anti_df['sr_index'] = anti_df['sr_index'].map(idx_dict)
    anti_df = anti_df.dropna()
    seq_df['id'] = seq_df['new_id']
    seq_df = seq_df.drop(columns=['new_id'])
    return seq_df, anti_df

# load_data
def load_data(args):
    seq_df = pd.read_csv(args.seq_path, sep=',')
    anti_df = pd.read_csv(args.anti_path, sep=',')
    # if args.filter:
    #     seq_df, anti_df = filter_data(seq_df, anti_df, args)

    if args.label == "NHT_distance":
        anti_df = anti_df[anti_df["NHT_distance"].notna()]
        labels = anti_df["NHT_distance"].to_numpy()
        at_ph_type = anti_df["NHT_at_ph"].tolist()
        sr_ph_type = anti_df["NHT_sr_ph"].tolist()
    elif args.label == "AHT_distance":
        anti_df = anti_df[anti_df["AHT_distance"].notna()]
        labels = anti_df["AHT_distance"].to_numpy()
        at_ph_type = anti_df["AHT_at_ph"].tolist()
        sr_ph_type = anti_df["AHT_sr_ph"].tolist()
    elif args.label == "NHT_class": 
        anti_df = anti_df[anti_df["NHT_class"].notna()]
        labels = anti_df["NHT_class"].to_numpy()
        at_ph_type = anti_df["NHT_at_ph"].tolist()
        sr_ph_type = anti_df["NHT_sr_ph"].tolist()
    elif args.label == "AHT_class":
        anti_df = anti_df[anti_df["AHT_class"].notna()]
        labels = anti_df["AHT_class"].to_numpy()
        at_ph_type = anti_df["AHT_at_ph"].tolist()
        sr_ph_type = anti_df["AHT_sr_ph"].tolist()

    seqs = seq_df['seq'].tolist()
    
    idx_to_seq = seq_df.set_index('index')['seq']
    idx_to_name = seq_df.set_index('index')['name']
    anti_df['at_seq'] = anti_df['at_index'].map(idx_to_seq)
    anti_df['sr_seq'] = anti_df['sr_index'].map(idx_to_seq)
    anti_df['at_name'] = anti_df['at_index'].map(idx_to_name)
    anti_df['sr_name'] = anti_df['sr_index'].map(idx_to_name)
    at_seqs = anti_df["at_seq"].tolist()
    sr_seqs = anti_df["sr_seq"].tolist()
    at_name = anti_df["at_name"].tolist()
    sr_name = anti_df["sr_name"].tolist()
    

    pair_year_large = np.maximum(anti_df['at_year'], anti_df['sr_year']).tolist()
    
    
    at_year = anti_df['at_index'].map(seq_df.set_index('index')['year']).tolist()
    sr_year = anti_df['sr_index'].map(seq_df.set_index('index')['year']).tolist()

    return at_seqs, sr_seqs, labels, at_year, sr_year, seqs, at_name, sr_name, at_ph_type, sr_ph_type

# spilt dataset for cross validation
def load_cv_idx(sample_num, args):
    train_idx_list = [[] for _ in range(args.cv)]
    val_idx_list = [[] for _ in range(args.cv)]
    test_idx_list = [[] for _ in range(args.cv)]
    for cv in range(0, args.cv):
        kf = KFold(n_splits=args.fold, random_state=cv, shuffle=True)
        for train_val_idx, test_idx in kf.split(range(sample_num)):
            train_idx, val_idx, _, _ = train_test_split(train_val_idx, train_val_idx, test_size=1 / (args.fold - 1), random_state=cv)
            
            train_idx_list[cv].append(train_idx.tolist())
            val_idx_list[cv].append(val_idx.tolist())
            test_idx_list[cv].append(test_idx.tolist())

    return train_idx_list, val_idx_list, test_idx_list