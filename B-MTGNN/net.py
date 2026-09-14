import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layer import mixprop, graph_constructor, dilated_inception, LayerNorm, series_decomp

fixed_seed=123

class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-5, init_min=0.1, init_max=0.3):
        super(ConcreteDropout, self).__init__()
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        init_min = math.log(init_min) - math.log(1. - init_min)
        init_max = math.log(init_max) - math.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.regularization = 0.0

    def forward(self, x, mc_dropout=False):
        p = torch.sigmoid(self.p_logit)
        
        weight_size = x.shape[1]
        
        dropout_regularizer = p * math.log(p + 1e-7) + (1. - p) * math.log(1. - p + 1e-7)
        dropout_regularizer *= self.dropout_regularizer * weight_size
        
        self.regularization = dropout_regularizer

        if self.training or mc_dropout:
            eps = 1e-7
            unif_noise = torch.rand_like(x)
            drop_prob = (math.log(p + eps) - math.log(1. - p + eps) + 
                         torch.log(unif_noise + eps) - torch.log(1. - unif_noise + eps))
            drop_prob = torch.sigmoid(drop_prob / 0.1)
            random_tensor = 1. - drop_prob
            retain_prob = 1. - p
            x = x * random_tensor / retain_prob
        
        return x

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN)
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str, target_idx=0):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_idx)
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")
        return x

    def _get_statistics(self, x):
        self.stdev = torch.sqrt(torch.var(x, dim=3, keepdim=True, unbiased=False) + self.eps).detach()
        
        if self.subtract_last:
            self.last = x[:, :, :, -1:].detach()
        else:
            self.mean = torch.mean(x, dim=3, keepdim=True).detach()

    def _normalize(self, x):
        shift = self.last if self.subtract_last else self.mean
        x = (x - shift) / self.stdev
        if self.affine:
            weight = self.affine_weight.view(1, -1, 1, 1)
            bias = self.affine_bias.view(1, -1, 1, 1)
            x = x * weight + bias
        return x

    def _denormalize(self, x, target_idx=0):
        stdev = self.stdev[:, target_idx:target_idx+1, :, :]
        shift = self.last[:, target_idx:target_idx+1, :, :] if self.subtract_last else self.mean[:, target_idx:target_idx+1, :, :]
        
        if self.affine:
            weight = self.affine_weight[target_idx].view(1, 1, 1, 1)
            bias = self.affine_bias[target_idx].view(1, 1, 1, 1)
            x = (x - bias) / (weight + self.eps)
            
        x = x * stdev + shift
        return x


class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, subtract_last=True, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 use_revin=True, use_decomp=True, use_concrete_dropout=True, use_ar_branch=True, use_afci_feedback=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        # Optional component switches; all components are enabled by default.
        self.use_revin = use_revin
        self.use_decomp = use_decomp
        self.use_concrete_dropout = use_concrete_dropout
        self.use_ar_branch = use_ar_branch
        self.use_afci_feedback = use_afci_feedback
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        
        self.revin = RevIN(num_features=in_dim, affine=False, subtract_last=subtract_last)
        
        self.decomp = series_decomp(kernel_size=25)
        self.concrete_drop = ConcreteDropout(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        
        # Dual head output for mu and sigma
        self.end_conv_2_mu = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2_sigma = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
                                             
        # AR Skip Connection based on Trend
        self.ar_layer_mu = nn.Linear(self.seq_length, out_dim)
        self.ar_layer_sigma = nn.Linear(self.seq_length, out_dim)

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)

    def _apply_dropout(self, x, training):
        """Concrete dropout when enabled, plain fixed-rate dropout otherwise.
        Sets self._reg to the regularisation contributed by this call."""
        if self.use_concrete_dropout:
            out = self.concrete_drop(x, mc_dropout=training)
            self._reg = self.concrete_drop.regularization
            return out
        self._reg = torch.zeros((), device=x.device, dtype=x.dtype)
        return F.dropout(x, p=self.dropout, training=training)

    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.use_revin:
            input = self.revin(input, mode='norm')

        # Ablating the autoregressive AFCI channel must happen AFTER RevIN.
        # _denormalize() rescales the output with the statistics of input
        # channel 0, so zeroing that channel beforehand destroys the output
        # scale rather than removing information from the encoder.
        if not self.use_afci_feedback:
            input = input.clone()
            input[:, 0, :, :] = 0.0
        
        # Series Decomposition
        if self.use_decomp:
            res_x, trend_x = self.decomp(input)
        else:
            res_x, trend_x = input, torch.zeros_like(input)

        if self.seq_length<self.receptive_field:
            res_x = nn.functional.pad(res_x,(self.receptive_field-self.seq_length,0,0,0))
            trend_x = nn.functional.pad(trend_x,(self.receptive_field-self.seq_length,0,0,0))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx) 
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(res_x)
        
        dropout_training = self.training or getattr(self, 'mc_dropout', False)
        
        skip_input = self._apply_dropout(res_x, dropout_training)
        skip = self.skip0(skip_input)
        
        self.reg_loss = self._reg

        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            
            x = self._apply_dropout(x, dropout_training)
            self.reg_loss += self._reg

            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.leaky_relu(skip, negative_slope=0.2)
        x = F.leaky_relu(self.end_conv_1(x), negative_slope=0.2)
        
        x_mu = self.end_conv_2_mu(x)
        x_sigma = self.end_conv_2_sigma(x)

        # AR Component on the Trend
        if self.use_ar_branch:
            ar_input = trend_x[:, 0, :, -self.seq_length:] # (batch_size, num_nodes, seq_length)
            ar_out_mu = self.ar_layer_mu(ar_input)
            ar_out_sigma = self.ar_layer_sigma(ar_input)

            ar_out_mu = ar_out_mu.permute(0, 2, 1).unsqueeze(3)
            ar_out_sigma = ar_out_sigma.permute(0, 2, 1).unsqueeze(3)

            x_mu = x_mu + ar_out_mu
            x_sigma = x_sigma + ar_out_sigma

        if self.use_revin:
            x_mu = self.revin(x_mu, mode='denorm', target_idx=0)
        
        # Ensure standard deviation is strictly positive FIRST in the normalized space
        x_sigma = F.softplus(x_sigma) + 1e-6

        # Sigma denorm (only scale, no shift) AFTER softplus
        if self.use_revin:
            stdev = self.revin.stdev[:, 0:1, :, :]
            if self.revin.affine:
                weight = self.revin.affine_weight[0].view(1, 1, 1, 1)
                x_sigma = x_sigma / (weight + self.revin.eps)
            x_sigma = x_sigma * stdev

        return x_mu, x_sigma
