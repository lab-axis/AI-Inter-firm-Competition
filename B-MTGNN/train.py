import ast
import argparse
import math
import time
import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib
import random
from o_util import DataLoaderS
from trainer import Optim
import sys
from random import randrange
from matplotlib import pyplot as plt
import time
from safetensors.torch import save_file
from safetensors import safe_open
import json


plt.rcParams['savefig.dpi'] = 1200  


def load_model(Data):
    with safe_open(args.o_save, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    arch_json = metadata.get("arch")
    if arch_json:
        arch = json.loads(arch_json)
        # 저장된 아키텍처로 모델 재생성
        model = gtnet(
            arch["gcn_true"],
            arch["buildA_true"],
            arch["gcn_depth"],
            arch["num_nodes"],
            device,
            Data.adj,
            dropout=arch["dropout"],
            subgraph_size=arch["subgraph_size"],
            node_dim=arch["node_dim"],
            dilation_exponential=arch["dilation_exponential"],
            conv_channels=arch["conv_channels"],
            residual_channels=arch["residual_channels"],
            skip_channels=arch["skip_channels"],
            end_channels=arch["end_channels"],
            seq_length=arch["seq_length"],
            in_dim=arch["in_dim"],
            out_dim=arch["out_dim"],
            layers=arch["layers"],
            propalpha=arch["propalpha"],
            tanhalpha=arch["tanhalpha"],
            layer_norm_affline=arch.get("layer_norm_affline", False),
        ).to(device)

        ret = model.load_state_dict(state_dict, strict=True)
        if hasattr(ret, "missing_keys") and ret.missing_keys:
            print("[load_state_dict] missing keys:", ret.missing_keys)
        if hasattr(ret, "unexpected_keys") and ret.unexpected_keys:
            print("[load_state_dict] unexpected keys:", ret.unexpected_keys)
    else:
        # 메타데이터가 없으면, 부분 로드(최소한 충돌은 방지)
        print("[warn] No arch metadata in checkpoint. Falling back to partial load (strict=False).")
        msd = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in msd and v.shape == msd[k].shape}
        msd.update(filtered)
        model.load_state_dict(msd, strict=False)
        print(f"[warn] Loaded {len(filtered)} / {len(msd)} tensors by shape-matching.")
    return model


def train(data, X, Y, model, criterion, optim, batch_size, cfg):
    model.train()  
    total_loss = 0  
    n_samples = 0  
    iter = 0  

    for X_batch, Y_batch in data.get_batches(X, Y, batch_size, True):  
        model.zero_grad()  
        X_batch = torch.unsqueeze(X_batch, dim=1)  
        X_batch = X_batch.transpose(2,3)  
        if iter % args.step_size == 0:  
            perm = np.random.permutation(range(args.num_nodes))  
        num_sub = int(args.num_nodes / args.num_split)  

        for j in range(args.num_split):  
            if j != args.num_split - 1:  
                id = perm[j * num_sub : (j + 1) * num_sub]  
            else:  
                id = perm[j * num_sub :]  

            id = torch.tensor(id).to(device)  
            tx = X_batch[:, :, :, :]  
            ty = Y_batch[:, :, :]  
            output = model(tx)  
            output = torch.squeeze(output, 3)  
            scale = data.scale.expand(output.size(0), output.size(1), data.m)  
            scale = scale[:, :, :]  
            output *= scale  
            ty *= scale  

            loss = criterion(output, ty)  
            loss.backward()  
            total_loss += loss.item()  
            n_samples += (output.size(0) * output.size(1) * data.m)  

            grad_norm = optim.step()  

        if iter % 1 == 0:  
            print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item()/(output.size(0) * output.size(1) * data.m)))  
        iter += 1  
    return total_loss / n_samples  


if __name__ == "__main__":  
    # === argparse 부분 주석 처리 ===  
    # parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    # parser.add_argument('--data', type=str, default='./data/sm_data.txt',
    #                     help='location of the data file')
    # parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
    #                     help='report interval')
    # parser.add_argument('--save', type=str, default='model/Bayesian/o_model.pt',
    #                     help='path to save the final model')
    # parser.add_argument('--optim', type=str, default='adam')
    # parser.add_argument('--L1Loss', type=bool, default=True)
    # parser.add_argument('--normalize', type=int, default=2)
    # parser.add_argument('--device',type=str,default='cuda:1',help='')
    # parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
    # parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
    # parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
    # parser.add_argument('--num_nodes',type=int,default=142,help='number of nodes/variables')
    # parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
    # parser.add_argument('--subgraph_size',type=int,default=20,help='k')
    # parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
    # parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
    # parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
    # parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
    # parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
    # parser.add_argument('--end_channels',type=int,default=64,help='end channels')
    # parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
    # parser.add_argument('--seq_in_len',type=int,default=10,help='input sequence length')
    # parser.add_argument('--seq_out_len',type=int,default=36,help='output sequence length')
    # parser.add_argument('--horizon', type=int, default=1) 
    # parser.add_argument('--layers',type=int,default=5,help='number of layers')
    # parser.add_argument('--batch_size',type=int,default=8,help='batch size')
    # parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    # parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
    # parser.add_argument('--clip',type=int,default=10,help='clip')
    # parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
    # parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')
    # parser.add_argument('--epochs',type=int,default=200,help='')
    # parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
    # parser.add_argument('--step_size',type=int,default=100,help='step_size')


    # args = parser.parse_args()
    # device = torch.device('cpu')
    # torch.set_num_threads(3)

    # 대신 config 로딩  
    from config import get_args  
    args = get_args()

    if torch.cuda.is_available():
        # GPU가 있다면
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    torch.set_num_threads(3)  

    def set_random_seed(seed):  
        random.seed(seed)  
        np.random.seed(seed)  
        torch.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False  

    fixed_seed = 123  
    set_random_seed(fixed_seed)  

    # read hyper-parameters  
    with open(args.hp_path, 'r') as file:  
        content = file.read()  
        hp = ast.literal_eval(content)  

    print('hp', hp)  

    # hyper-parameters unpack  
    gcn_depth = hp[0]  
    lr = hp[1]  
    conv = hp[2]  
    res = hp[3]  
    skip = hp[4]  
    end = hp[5]  
    layer = hp[-2]  
    k = hp[6]  
    dropout = hp[7]  
    dilation_ex = hp[8]  
    node_dim = hp[9]  
    prop_alpha = hp[10]  
    tanh_alpha = hp[11]  
    epochs = hp[-1]  

    Data = DataLoaderS(
        args.data, 
        args.train_ratio, 
        args.valid_ratio, 
        device, 
        args.horizon, 
        args.seq_in_len, 
        args.graph_file,
        args.normalize, 
        args.seq_out_len
    )
    model = load_model(Data)

    print(args)  
    print('The receptive field size is', model.receptive_field)  
    nParams = sum([p.nelement() for p in model.parameters()])  
    print('Number of model parameters is', nParams, flush=True)  

    if args.L1Loss:  
        criterion = nn.L1Loss(reduction='sum').to(device)  
    else:  
        criterion = nn.MSELoss(reduction='sum').to(device)  
    evaluateL2 = nn.MSELoss(reduction='sum').to(device)  
    evaluateL1 = nn.L1Loss(reduction='sum').to(device)  

    optim = Optim(  
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay  
    )  

    try:  
        print('begin training')  
        for epoch in range(1, epochs + 1):  
            print('epoch:', epoch)  
            epoch_start_time = time.time()  
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size, args)  

        arch_meta = {
            "gcn_true": args.gcn_true,
            "buildA_true": args.buildA_true,
            "gcn_depth": gcn_depth,
            "num_nodes": args.num_nodes,
            "dropout": dropout,
            "subgraph_size": k,
            "node_dim": node_dim,
            "dilation_exponential": dilation_ex,
            "conv_channels": conv,
            "residual_channels": res,
            "skip_channels": skip,
            "end_channels": end,
            "seq_length": args.seq_in_len,
            "in_dim": args.in_dim,
            "out_dim": args.seq_out_len,
            "layers": layer,
            "propalpha": prop_alpha,
            "tanhalpha": tanh_alpha,
            "layer_norm_affline": False  # net.py에서 그대로 사용
        }

        metadata = {"arch": json.dumps(arch_meta)}

        # state_dict 저장
        state = model.state_dict()
        save_file(state, args.save, metadata=metadata)

    except KeyboardInterrupt:  
        print('-' * 89)  
        print('Exiting from training early')  

