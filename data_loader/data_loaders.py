from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import math
import torch
import random
import numpy as np
import pandas as pd
import tarfile
from copy import deepcopy
from base import BaseDataLoader
from utils import get_corresponding_values, NewParser, collate_fn, infer_routing_geant, infer_routing_GBN, infer_routing_nsf3, ned2lists, load_routing, make_paths, make_indices


# transform = {"type": "original", "mean_TM": 0.5,"std_TM":0.5,
#              "mean_delay":2.8,"std_delay":2.5,"scale_drops":12000,"bias_drops":0.5,
#              "mean_jitter":2.0,"std_jitter":2.0}
# "transform": {"mean_TM": 0.475,"std_TM":0.259,"mean_delay":1.037,"std_delay":1.673,"mean_drops":315.725,"std_drops":1137.143,"mean_jitter":0.250,"std_jitter":0.422}
        # transform={"mean_TM": 0.558, "std_TM": 0.295, 
        #    "mean_delay": 0.141, "std_delay": 0.887, 
        #    "mean_link_capacity": 14.286, "std_link_capacity": 10.498}

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
        self.feature_folders = [os.path.join(self.dataset_dir, name) for name in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, name))]
        ## Network configuration path
        self.network_configuration_path = os.path.join(data_dir, "Network_" + base_folder + ".ned")
        self.transform = transform
        ## store the dataset in lists
        self.dataset_traffic = []
        self.dataset_delay = []
        self.dataset_jitter = []
        self.dataset_link_capacity = []
        self.dataset_link_indices = []
        self.dataset_path_indices = []
        self.dataset_sequ_indices = []
        self.dataset_n_paths = []
        self.dataset_n_links = []
        self.dataset_n_total = []
        self.dataset_paths = []
        for feature_file in self.feature_folders:
            delay_file = open(os.path.join(feature_file, "simulationResults.txt"), "r").readlines()
            routing_file = os.path.join(feature_file, "Routing.txt")
            con, n, link_cap = ned2lists(self.network_configuration_path)
            posParser = NewParser(n)
            R = load_routing(routing_file)
            paths, link_capacities = make_paths(R, con, link_cap)
            link_indices, path_indices, sequ_indices = make_indices(paths)
            n_paths = len(paths)
            n_links = max(max(paths)) + 1
            a = np.zeros(n_paths)
            d = np.zeros(n_paths)
            j = np.zeros(n_paths)
            as_ = []
            ds_ = []
            js_ = []
            for line in delay_file:
                # line = line.decode().split(',')
                line = line.rstrip('\n').split(',')
                get_corresponding_values(posParser, line, n, a, d, j)
                as_.append(a)
                ds_.append(d)
                js_.append(j)
            as_ = np.stack(as_)
            ds_ = np.stack(ds_)
            js_ = np.stack(js_)
            link_capacities = np.array(link_capacities)
            if self.transform != None:
                if "mean_TM" in self.transform.keys() and "std_TM" in self.transform.keys():
                    as_ = (as_ - self.transform["mean_TM"]) / self.transform["std_TM"]
                if "mean_delay" in self.transform.keys() and "std_delay" in self.transform.keys():
                    ds_ = (ds_ - self.transform["mean_delay"]) / self.transform["std_delay"]
                if "mean_jitter" in self.transform.keys() and "std_jitter" in self.transform.keys():
                    js_ = (js_ - self.transform["mean_jitter"]) / self.transform["std_jitter"]
                if "mean_link_capacity" in self.transform.keys() and "std_link_capacity" in self.transform.keys():
                    link_capacities = (link_capacities - self.transform["mean_link_capacity"]) / self.transform["std_link_capacity"]
            self.dataset_traffic.append(as_)
            self.dataset_delay.append(ds_)
            self.dataset_jitter.append(js_)
            self.dataset_link_indices.append(link_indices)
            self.dataset_path_indices.append(path_indices)
            self.dataset_sequ_indices.append(sequ_indices)
            self.dataset_n_paths.append(n_paths)
            self.dataset_n_links.append(n_links)
            self.dataset_n_total.append(n_links)
            self.dataset_paths.append(paths)
            self.dataset_link_capacity.append(link_capacities)
    
    def __len__(self):
        length = 0
        for traffic in self.dataset_traffic:
            length += traffic.shape[0]
        return length

    def __getitem__(self, idx):
        file_index = 0
        row_index = 0
        total_length = 0
        while True:
            file_length = self.dataset_traffic[file_index].shape[0]
            if file_length + total_length >= idx + 1:
                row_index = idx - total_length
                break
            else:
                file_index += 1
                total_length += file_length
        TM = self.dataset_traffic[file_index]
        delay = self.dataset_delay[file_index]
        jitter = self.dataset_jitter[file_index]
        link_capacity = self.dataset_link_capacity[file_index]
        link_indices = self.dataset_link_indices[file_index]
        path_indices = self.dataset_path_indices[file_index]
        sequ_indices = self.dataset_sequ_indices[file_index]
        n_paths = self.dataset_n_paths[file_index]
        n_links = self.dataset_n_links[file_index]
        n_total = self.dataset_n_total[file_index]
        paths = self.dataset_paths[file_index]
        delay = delay[row_index, :]
        TM = TM[row_index, :]
        jitter = jitter[row_index, :]
        targets_dict = {}
        targets_dict["delay"] = delay
        targets_dict["jitter"] = jitter
        targets_lists = [targets_dict[i] for i in targets_dict.keys() if i in self.prediction_targets]
        targets = np.stack(targets_lists, axis=0)
        
        return (TM, link_capacity, link_indices, path_indices, sequ_indices, n_paths, n_links, n_total, paths), targets

class ConcatNetDataset(Dataset):
    """
    Network data loading 
    """
    def __init__(self, data_dir, prediction_targets, datasets, transform=None):
        self.datasets = []
        self.dataset_length = []
        for dataset in datasets:
            #print(os.path.join(data_dir,dataset))
            netDataset=NetDataset(os.path.join(data_dir,dataset), ["delay"], transform=transform)
            self.datasets.append(netDataset)
            self.dataset_length.append(len(netDataset))
            
    def __len__(self):
        return sum(self.dataset_length)
    
    def __getitem__(self, idx):
        file_index = 0
        row_index = 0
        total_length = 0
        while True:
            file_length = self.dataset_length[file_index]
            if file_length + total_length >= idx + 1:
                row_index = idx - total_length
                break
            else:
                file_index += 1
                total_length += file_length
        data, targets = self.datasets[file_index][row_index]
        return data, targets

class NetDataLoader(BaseDataLoader):
    """
    Network data loading 
    """

    def __init__(self, data_dir, prediction_targets, datasets, batch_size, shuffle=True, validation_split=0.0, num_workers=1, transform=None):
        self.data_dir = data_dir
        self.dataset = ConcatNetDataset(self.data_dir, prediction_targets, datasets, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)
    
