import pickle
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import sys
import csv
from collections import defaultdict

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataLoaderS(object):
    # train and valid are retained for interface compatibility; split_policy defines windows.
    def __init__(self, file_name, train, valid, device, horizon, window, graph_file, normalize=2, out=1, nodes_file=None,
                 split_policy="legacy", test_reserve=24, valid_span=24, num_eval=7):
        self.P = window
        self.h = horizon

        if file_name.endswith('.npy'):
            # Load 3D numpy array: [num_nodes, seq_length, num_features]
            self.rawdat = np.load(file_name)
            # Transpose to [seq_length, num_nodes, num_features]
            self.rawdat = self.rawdat.transpose(1, 0, 2)
            
            # Generate actual monthly dates starting from 2014-01-01
            start_date = '2014-01-01'
            self.timeindex = pd.date_range(start=start_date, periods=self.rawdat.shape[0], freq='MS').strftime('%Y-%m-%d').tolist()
            
            # Try to load firm names from nodes_file
            if nodes_file and os.path.exists(nodes_file):
                try:
                    node_df = pd.read_csv(nodes_file)
                    self.col = node_df['firm_id'].tolist()
                    print(f"Loaded {len(self.col)} firm names from {nodes_file}")
                except:
                    self.col = [f"Node{i}" for i in range(self.rawdat.shape[1])]
            else:
                self.col = [f"Node{i}" for i in range(self.rawdat.shape[1])]
        else:
            df_raw = pd.read_csv(file_name, index_col=0)
            self.rawdat = df_raw.values
            self.col = list(df_raw.columns)
            self.timeindex = list(df_raw.index)

        self.n, self.m, self.f = self.rawdat.shape # Time, Nodes, Features
        self.shift = 0
        self.dat = np.zeros(self.rawdat.shape)
        self.normalize = normalize
        self.out_len = out

        # Obtain target windows and the scaling boundary from split_policy.build().
        try:
            import split_policy as _sp
        except ImportError:
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import split_policy as _sp
        self.split_plan = _sp.build(n=self.n, P=self.P, h=self.h,
                                    out=self.out_len, policy=split_policy,
                                    num_eval=num_eval,
                                    test_reserve=test_reserve,
                                    valid_span=valid_span)
        print(_sp.report(self.split_plan), flush=True)

        # Scale for each node and each feature
        self.scale = np.ones((self.m, self.f))
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n))

        self.scale = torch.from_numpy(self.scale).float().to(device)
        self.device = device
        if graph_file:
            raise ValueError('External adjacency is disabled in this release.')
        self.adj = None  # gtnet constructs adjacency from its trained embeddings.

    def _normalized(self, normalize):
        # Use the normalization boundary returned by the selected split policy.
        train_end = self.split_plan["train_end"]
        
        if (normalize == 0):
            self.dat = self.rawdat
        elif (normalize == 1):
            mx = np.max(np.abs(self.rawdat[:train_end]))
            if mx == 0: mx = 1.0
            self.dat = self.rawdat / mx
        elif (normalize == 2):
            for i in range(self.m): # Nodes
                for j in range(self.f): # Features
                    mx = np.max(np.abs(self.rawdat[:train_end, i, j]))
                    if mx == 0: mx = 1.0
                    self.scale[i, j] = mx
                    self.dat[:, i, j] = self.rawdat[:, i, j] / mx

    def _split(self, train, valid):
        out = self.out_len

        # Construct samples from the origin lists returned by split_policy.build().
        test_starts  = list(self.split_plan["test_starts"])
        valid_starts = list(self.split_plan["valid_starts"])
        train_starts = list(self.split_plan["train_starts"])

        self.train_starts = train_starts

        self.train = self._batchify(train_starts)
        self.valid = self._batchify(valid_starts)
        self.test  = self._batchify(test_starts)

        self.test_window    = torch.from_numpy(self.dat[-(out + self.P):, :])
        self.test_window_tf = self.timeindex[-(out + self.P):]

        self.valid_window    = torch.from_numpy(self.dat[-(2*out + self.P):-(out), :])
        self.valid_window_tf = self.timeindex[-(2*out + self.P):-(out)]


    def _batchify(self, start_indices):
        n = len(start_indices) 
        X = torch.zeros((n, self.P, self.m, self.f))
        Y = torch.zeros((n, self.out_len, self.m, self.f)) 

        tf = []
        for i in range(n): 
            start_idx = start_indices[i]
            end = start_idx - self.h + 1 
            start = end - self.P 
            X[i, :, :, :] = torch.from_numpy(self.dat[start:end, :, :]) 
            Y[i, :, :, :] = torch.from_numpy(self.dat[start_idx:start_idx+self.out_len, :, :])
            tf.append(self.timeindex[start_idx:start_idx+self.out_len])
            
        return [X, Y, tf]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

    #by Zaid et al.
    # returns column names within dataset    
    def create_columns(self):

        file_name='data_/data/data.csv'
        if self.m==123:
            file_name='data/sm_data_g.csv'

        # Read the CSV file of the dataset
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            # Read the first row
            col = [c for c in next(reader)]
            
            if 'Date' in col[0]:
                return col[1:]
            
            return col

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """Batch paired arrays, optionally repeating the final sample to fill the last batch."""
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """Standardize inputs using the supplied mean and standard deviation."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """Return the symmetric normalized Laplacian I - D^(-1/2) A D^(-1/2)."""
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data



def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    z = torch.tensor((x-mean)/std,dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))
