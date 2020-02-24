import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from base import BaseModel

class GraphNeuralNet(BaseModel):
    def __init__(self, link_state_dim, path_state_dim, fc1_dim, fc2_dim, output_dim, dropout_rate, T, link_capacity):
        super().__init__()
        self.link_state_dim = link_state_dim
        self.path_state_dim = path_state_dim
        self.dropout_rate = dropout_rate
        self.T = T
        self.link_capacity = link_capacity
        self.edge_update = nn.GRU(input_size=self.path_state_dim, hidden_size=self.link_state_dim, batch_first=True)
        self.path_update = nn.GRU(input_size=self.link_state_dim, hidden_size=self.path_state_dim, batch_first=True) 
        self.fc1 = nn.Linear(self.path_state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, output_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        ## initialization of the path and link hiddent states
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        features, list_link_indices, list_path_indices, list_sequ_indices, n_paths, n_links, n_total, paths = x
        batch_size, feature_dim = features.shape
        path_state = torch.zeros(size=(batch_size, self.path_state_dim)).to(device)
        link_state = torch.zeros(size=(torch.sum(n_links), self.link_state_dim)).to(device)
        link_state[:, 0] = torch.tensor(self.link_capacity).to(device)
        path_state[:, :feature_dim] = features
        ## start of the message passing
        for _ in range(self.T):
            flat_list = [torch.index_select(link_state, 0, path, out=None).to(device) for path in paths]
            padded_sequence = pad_sequence(flat_list, batch_first=True)
            message, path_hidden_states = self.path_update(padded_sequence, path_state.unsqueeze(0))
            path_state = path_hidden_states.squeeze()
            message_accumulating = message[[list_path_indices, list_sequ_indices]]
            message_passing = torch.zeros(size=(torch.sum(n_links), self.path_state_dim)).to(device)
            message_passing = message_passing.index_add(0, list_link_indices, message_accumulating)
            _, link_hidden_states = self.edge_update(message_passing.unsqueeze(1), link_state.unsqueeze(0))
            link_state = link_hidden_states.squeeze()
        ## generate final prediction
        predictions = self.dropout(F.selu(self.fc1(path_state)))
        predictions = self.dropout(F.selu(self.fc2(predictions)))
        predictions = self.fc3(predictions)

        return predictions