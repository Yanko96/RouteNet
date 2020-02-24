from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import math
import torch
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from base import BaseDataLoader
from utils import collate_fn, infer_routing_geant, infer_routing_GBN, infer_routing_nsf3, genPath, pairwise, ned2lists, extract_links, load_and_process, load, load_routing, make_paths, make_indices


# transform = {"type": "original", "mean_TM": 0.5,"std_TM":0.5,
#              "mean_delay":2.8,"std_delay":2.5,"scale_drops":12000,"bias_drops":0.5,
#              "mean_jitter":2.0,"std_jitter":2.0}
# "transform": {"mean_TM": 0.475,"std_TM":0.259,"mean_delay":1.037,"std_delay":1.673,"mean_drops":315.725,"std_drops":1137.143,"mean_jitter":0.250,"std_jitter":0.422}
        


class NetDataset(Dataset):
    """
    Network data loading 
    """

    def __init__(self, data_dir, prediction_targets, transform=None):
        
        """
        Reads the dataset in memory
        
        Inputs:
        data_dir: the directory to the dataset
        prediction_targets: what features do you want your model to predict
        transform: a dictionary contains the mean and the std of the features 
        """
        
        base_folder = os.path.basename(data_dir)
        self.dataset_name = base_folder
        self.dataset_dir=data_dir
        self.prediction_targets = prediction_targets
        feature_path = os.path.join(data_dir, "delays" + base_folder.title())
        unchecked_feature_files = os.listdir(feature_path)
        self.feature_files = []
        ## find the feature files
        for feature_file in unchecked_feature_files:
            if feature_file.endswith(".txt"):
                self.feature_files.append(feature_file)
        ## Network configuration path
        self.network_configuration_path = os.path.join(data_dir, "Network" + base_folder.title() + ".ned")
        self.transform = transform
        ## store the dataset in lists
        self.dataset_TM = []
        self.dataset_delay = []
        self.dataset_jitter = []
        self.dataset_drops = []
        self.dataset_link_indices = []
        self.dataset_path_indices = []
        self.dataset_sequ_indices = []
        self.dataset_n_paths = []
        self.dataset_n_links = []
        self.dataset_n_total = []
        self.dataset_paths = []
        self.feature_file_lengths = []
        ## loop over feature files, each one contains 500 data (simulations)
        for feature_file in self.feature_files:
            feature_file = os.path.join(self.dataset_dir, "delays" + self.dataset_name.title(), feature_file)
            self.feature_file_lengths.append(len(open(feature_file, "r").readlines()))
            if self.dataset_name == "nsfnet":
                routing_file = infer_routing_nsf3(feature_file)
            elif self.dataset_name == "geant2":
                routing_file = infer_routing_geant(feature_file)
            elif self.dataset_name == "GBN":
                routing_file = infer_routing_GBN(feature_file)
            ## reads the network topology and the routing schemes using the methods in utilities
            con,n = ned2lists(self.network_configuration_path)
            Global, TM_index, delay_index, jitter_index, drop_index = load(feature_file, n, True)
            R = load_routing(routing_file)
            paths = make_paths(R, con)
            link_indices, path_indices, sequ_indices = make_indices(paths)
            delay = Global.take(delay_index, axis=1)   
            n_paths = delay.shape[1]
            TM = Global.take(TM_index, axis=1)
            jitter =  Global.take(jitter_index, axis=1)
            drops =  Global.take(drop_index, axis=1)
            n_links = max(max(paths)) + 1
            n_total = len(path_indices)      
            if self.transform != None:
                if self.transform["type"] == "normal":
                    if "mean_TM" in self.transform.keys() and "std_TM" in self.transform.keys():
                        TM = pd.DataFrame((TM.values-self.transform["mean_TM"])/self.transform["std_TM"])
                    if "mean_delay" in self.transform.keys() and "std_delay" in self.transform.keys():
                        delay = pd.DataFrame((delay.values-self.transform["mean_delay"])/self.transform["std_delay"]) 
                    if "mean_jitter" in self.transform.keys() and "std_jitter" in self.transform.keys():
                        jitter = pd.DataFrame((jitter.values-self.transform["mean_jitter"])/self.transform["std_jitter"]) 
                    if "mean_drops" in self.transform.keys() and "std_drops" in self.transform.keys():
                        drops = pd.DataFrame((drops.values-self.transform["mean_drops"])/self.transform["std_drops"]) 
                if self.transform["type"] == "original":
                    if "mean_TM" in self.transform.keys() and "std_TM" in self.transform.keys():
                        TM = pd.DataFrame((TM.values-self.transform["mean_TM"])/self.transform["std_TM"])
                    if "mean_delay" in self.transform.keys() and "std_delay" in self.transform.keys():
                        delay = pd.DataFrame((delay.values-self.transform["mean_delay"])/self.transform["std_delay"]) 
                    if "mean_jitter" in self.transform.keys() and "std_jitter" in self.transform.keys():
                        jitter = pd.DataFrame((np.log(jitter.values)-self.transform["mean_jitter"])/self.transform["std_jitter"]) 
                    if "scale_drops" in self.transform.keys() and "bias_drops" in self.transform.keys():
                        drops = pd.DataFrame((drops.values/self.transform["scale_drops"])/(0.5*TM.values+self.transform["bias_drops"])) 
            self.dataset_TM.append(TM)
            self.dataset_delay.append(delay)
            self.dataset_jitter.append(jitter)
            self.dataset_drops.append(drops)
            self.dataset_link_indices.append(link_indices)
            self.dataset_path_indices.append(path_indices)
            self.dataset_sequ_indices.append(sequ_indices)
            self.dataset_n_paths.append(n_paths)
            self.dataset_n_links.append(n_links)
            self.dataset_n_total.append(n_total)
            self.dataset_paths.append(paths)      


    def __len__(self):
        length = 0
        for feature_file in self.feature_files:
            length += len(open(os.path.join(self.dataset_dir, "delays" + self.dataset_name.title(), feature_file), "r").readlines())
        return length

    def __getitem__(self, idx):
        file_index = 0
        row_index = 0
        total_length = 0
        while True:
            file_length = self.feature_file_lengths[file_index]
            if file_length + total_length >= idx + 1:
                row_index = idx - total_length
                break
            else:
                file_index += 1
                total_length += file_length
        TM = self.dataset_TM[file_index]
        delay = self.dataset_delay[file_index]
        jitter = self.dataset_jitter[file_index]
        drops = self.dataset_drops[file_index]
        link_indices = self.dataset_link_indices[file_index]
        path_indices = self.dataset_path_indices[file_index]
        sequ_indices = self.dataset_sequ_indices[file_index]
        n_paths = self.dataset_n_paths[file_index]
        n_links = self.dataset_n_links[file_index]
        n_total = self.dataset_n_total[file_index]
        paths = self.dataset_paths[file_index]
        delay = delay.iloc[row_index].values
        TM = TM.iloc[row_index].values
        jitter = jitter.iloc[row_index].values
        drops = drops.iloc[row_index].values
        feature_dict={}
        feature_dict["TM"] = TM
        feature_dict["delay"] = delay
        feature_dict["jitter"] = jitter
        feature_dict["drops"] = drops
        feature_names = ["TM", "delay", "jitter", "drops"]
        targets_list = []
        for target in self.prediction_targets:
            targets_list.append(feature_dict[target])
            feature_dict.pop(target)
            feature_names.remove(target)
        # if len(targets_list)>1:
        targets = np.stack(targets_list, axis=0)
        # else:
        #     targets = targets_list[0]
        feature_lists = [feature_dict[i] for i in feature_names]
        features = np.stack(feature_lists, axis=0)
        
        return (features, link_indices, path_indices, sequ_indices, n_paths, n_links, n_total, paths), targets

class NetDataLoader(BaseDataLoader):
    """
    Network data loading 
    """

    def __init__(self, data_dir, prediction_targets, batch_size, shuffle=True, validation_split=0.0, num_workers=1, transform=None):
        self.data_dir = data_dir
        self.dataset = NetDataset(self.data_dir, prediction_targets, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)
    