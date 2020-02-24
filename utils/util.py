import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
import pandas as pd
import networkx as nx
import itertools as it
import collections 
import re
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.init as init

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

def collate_fn(batch):
    data, targets = zip(*batch)
    features = [datum[0] for datum in data]
    link_indices = [datum[1] for datum in data]
    path_indices = [datum[2] for datum in data]
    sequ_indices = [datum[3] for datum in data]
    n_paths = [datum[4] for datum in data]
    n_links = [datum[5] for datum in data]
    n_total = [datum[6] for datum in data]
    paths = [datum[7] for datum in data]
    features_tensor = torch.cat([torch.tensor(feature, dtype=torch.float32) for feature in features], dim=1).permute(1, 0)
    targets_tensor = torch.cat([torch.tensor(target, dtype=torch.float32) for target in targets], dim=1).permute(1, 0)
    list_sequ_indices = torch.tensor([item for sublist in sequ_indices for item in sublist])
    length = 0
    list_path_indices=[]
    for i, sublist in enumerate(path_indices):
        for item in sublist:
            list_path_indices.append(item+length)
        length += n_paths[i]
    length = 0
    list_link_indices=[]
    for i, sublist in enumerate(link_indices):
        for item in sublist:
            list_link_indices.append(item+length)
        length += n_links[i]
    length = 0
    list_paths = []
    for i, sublist in enumerate(paths):
        for subsublist in sublist:
            tmp = [item + length for item in subsublist] 
            list_paths.append(tmp)
        length += n_links[i]
    list_link_indices = torch.tensor(list_link_indices, dtype=torch.long)
    list_path_indices = torch.tensor(list_path_indices, dtype=torch.long)
    list_sequ_indices = torch.tensor(list_sequ_indices, dtype=torch.long)
    n_paths = torch.tensor(n_paths, dtype=torch.long)
    n_links = torch.tensor(n_links, dtype=torch.long)
    n_total = torch.tensor(n_total, dtype=torch.long)
    for i in range(len(list_paths)):
        list_paths[i] = torch.tensor(list_paths[i], dtype=torch.long)
    
    return (features_tensor, list_link_indices, list_path_indices, list_sequ_indices, n_paths, n_links, n_total, list_paths), targets_tensor

def infer_routing_GBN(data_file):
    rf=re.sub(r'dGlobal_\d+_\d+_','Routing_', data_file).\
    replace('delay','routing')
    return rf

def infer_routing_geant(data_file):
    rf=re.sub(r'dGlobal_G_\d+_\d+_','RoutingGeant2_', data_file).\
    replace('delaysGeant2','routingsGeant2')
    return rf

def infer_routing_nsf3(data_file):
    rf=re.sub(r'dGlobal_\d+_\d+_','Routing_', data_file).\
    replace('delaysNsfnet','routingNsfnet')
    return rf

def genPath(R,s,d,connections):
    ## generate a path from the source to the destination 
    while s != d:
        yield s
        s = connections[s][R[s,d]]
    yield s

def pairwise(iterable):
    ## s -> (s0,s1), (s1,s2), (s2, s3), ...
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


def ned2lists(fname):
    ## convert .ned file into network lists
    ## returns a connectivity list l where l[n] shows the nodes that have connection with node n 
    ## and number of nodes
    channels = []
    with open(fname) as f:
        p = re.compile(r'\s+node(\d+).port\[(\d+)\]\s+<-->\s+Channel\s+<-->\s+node(\d+).port\[(\d+)\]')
        for line in f:
            m=p.match(line)
            if m:
                #print(line, m.groups())
                channels.append(list(map(int,m.groups())))
    n=max(map(max, channels))+1
    connections = [{} for i in range(n)]
    for c in channels:
        connections[c[0]][c[1]]=c[2]
        connections[c[2]][c[3]]=c[0]
    connections = [[v for k,v in sorted(con.items())] 
                   for con in connections ]
    return connections,n


def extract_links(n, connections):
    ## A - Adjacency matrix
    ## create a graph using networkX & A and get edges from built-in methods
    A = np.zeros((n,n))

    for a,c in zip(A,connections):
        a[c]=1

    G=nx.from_numpy_array(A, create_using=nx.DiGraph())
    edges=list(G.edges)
    return edges

def load_and_process(routing_file, data_file,edges,connections,n=15):
    ## R is routing file 
    ## features are the traffic data and labels are delays
    R=np.loadtxt(routing_file, dtype=np.int32)
    data = np.loadtxt(data_file)
    traffic = np.reshape(data[:,0:n*n],(-1,n,n))   # bandwidth
    delay = np.reshape(data[:,n*n:2*n*n],(-1,n,n))
    #packet_loss = data[:,-1]
    
    paths=[]
    features=[]
    labels=[]

    for i in range(n):
        for j in range(n):
            if i != j:
                paths.append([edges.index(tup) for tup in pairwise(genPath(R,i,j,connections))])  # path generation according to R 
                features.append(traffic[:,i,j])
                labels.append(delay[:,i,j])
    features = np.stack(features).T
    labels = np.stack(labels).T
    return paths,features,labels   # path bandwidth delay 

def load(data_file,n, full=False):
    ## read the delay .txt data in the dataset 
    ## Output: Global: all the data in the dataset
    ## TM_index: index of bandwith data n*n
    ## delay_index: index of 7 link features
    ## ... and so on
    names=[]

    TM_index=[]
    delay_index=[]
    if full:
        jitter_index=[]
        drop_index=[]

    counter=0
    for i in range(n):
        for j in range(n):
            names.append('a{}_{}'.format(i,j))
            if i != j:
                TM_index.append(counter)
            counter += 1
    for i in range(n):
        for j in range(n):
            for k in ['average', 'q10','q20','q50','q80','q90','variance']:
                names.append('delay{}_{}_{}'.format(i,j,k))
                if i != j and k == 'average':
                    delay_index.append(counter)
                if i != j and k == 'variance' and full:
                    jitter_index.append(counter)
                counter += 1
    for i in range(n):
        for j in range(n):
            names.append('drop{}_{}'.format(i,j))
            if full and i != j:
                drop_index.append(counter)
            counter += 1
    names.append('empty')
            
    Global=pd.read_csv(data_file ,header=None, names=names,index_col=False)
    if full:
        return Global, TM_index, delay_index, jitter_index, drop_index
    else:
        return Global, TM_index, delay_index

def load_routing(routing_file):
    ## just read the route matrix in numpy array 
    R = pd.read_csv(routing_file, header=None, index_col=False)
    R=R.drop([R.shape[0]], axis=1)
    return R.values
   

def make_paths(R,connections):
    ## generate edge index for each path (n*n-n)
    n = R.shape[0]
    edges = extract_links(n, connections)
    paths=[]
    for i in range(n):
        for j in range(n):
            if i != j:
                paths.append([edges.index(tup) for tup in pairwise(genPath(R,i,j,connections))])
    return paths

def make_indices(paths):
    link_indices=[]        ## a long list that contains the same elements as paths but instead of a nested list 
    path_indices=[]        ## indicates which paths the elements in link_indices belong to 
    sequ_indices=[]        ## indicates the order of the element in link_indices in the path it belongs to 
    segment=0
    for p in paths:
        link_indices += p    # as defined in make_paths()
        path_indices += len(p)*[segment]                 
        sequ_indices += list(range(len(p)))              # [0, 1, 2, 3, ... len(p)]
        segment +=1
    return link_indices, path_indices, sequ_indices

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
