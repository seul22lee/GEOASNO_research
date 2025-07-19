import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.fft
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
import csv
import operator
from functools import reduce
from functools import partial
import os
from timeit import default_timer
from utilities4 import *
import sys
from VAE import Encoder, Decoder, VAE_model
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.fft
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

import operator
from functools import reduce
from functools import partial
import os
from timeit import default_timer
from utilities4 import *
import sys
from VAE import Encoder, Decoder, VAE_model
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
from timeit import default_timer
from itertools import product

# Filename for saving the model
# ────────────────────────────────────────────────────────────────────────────────
# 1. Ablation Study Configurations
# ────────────────────────────────────────────────────────────────────────────────
# 1) Define the grid of values for each hyperparameter
learning_rates = [1e-4, 1e-3, 1e-2]
weight_decays  = [1e-4, 1e-3, 1e-2]
gammas         = [0.3, 0.5, 0.7]

# 2) Build the full‐factorial list of configs
ablation_configs = [
    {'learning_rate': lr, 'weight_decay': wd, 'gamma': γ}
    for lr, wd, γ in product(learning_rates, weight_decays, gammas)
]

base_output_dir = "./ablation_results"
os.makedirs(base_output_dir, exist_ok=True)
for cfg in ablation_configs:
    name = f"lr_{cfg['learning_rate']}_wd_{cfg['weight_decay']}_gamma_{cfg['gamma']}"
    cfg['output_dir'] = os.path.join(base_output_dir, name)
    os.makedirs(cfg['output_dir'], exist_ok=True)


# Transformer Encoder model definition
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        # Linear layer to embed the input data to a higher dimension
        self.embedding = nn.Linear(input_dim, embed_dim)
        # Positional encoding as a learnable parameter
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Assuming a max length of 50
        # Define a single transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        # Stack multiple transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final fully connected layer to map to the output dimension
        self.fc = nn.Linear(embed_dim, output_dim)  # Reduce to (output_dim / seq_len)

    def forward(self, x):
        # Apply embedding and add positional encoding
        x = self.embedding(x) + self.positional_encoding
        # Pass through the transformer encoder
        x = self.transformer_encoder(x)
        # Apply the fully connected layer
        x = self.fc(x)  # Pooling: mean across time steps
        return x
    
    # Initializing machine learning weigths 
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

# MS_Loss model definition
class MS_Loss(nn.Module):
    def __init__(self, tokend, r, dk, nb, featured, out_dim=1):
        super(MS_Loss, self).__init__()
        # Initialize model parameters
        self.tokend = tokend
        self.r = r
        self.dk = dk
        self.nb = nb
        self.featured = featured
        self.out_dim = out_dim

        # Define layers in the model using nested loops
        for i in range(self.r):
            for j in range(self.nb):
                self.add_module('fcq_%d_%d' % (j, i), nn.Linear(featured, self.dk))
                self.add_module('fck_%d_%d' % (j, i), nn.Linear(featured, self.dk))
            self.add_module('fcp%d' % i, nn.Linear(self.tokend, out_dim))

        # Layer normalization
        for j in range(self.nb + 1):
            self.add_module('fcn%d' % j, nn.LayerNorm([self.tokend, featured]))

    def forward(self, xy):
        # Softmax activation for attention mechanism
        m = nn.Softmax(dim=2)
        batchsize = xy.shape[0]
        Vinput = self._modules['fcn%d' % 0](xy)
        out_ft = torch.zeros((batchsize, self.featured, self.out_dim), device=xy.device)
        for j in range(self.nb - 1):
            mid_ft = torch.zeros((batchsize, self.tokend, self.featured), device=xy.device)
            for i in range(self.r):
                Q = self._modules['fcq_%d_%d' % (j, i)](Vinput)
                K = self._modules['fck_%d_%d' % (j, i)](Vinput)
                Attn = m(torch.matmul(Q, torch.transpose(K, 1, 2)) / torch.sqrt(torch.tensor(self.dk, dtype=torch.float)))
                V = (torch.matmul(Attn, Vinput))
                mid_ft = mid_ft + V
            Vinput = self._modules['fcn%d' % (j + 1)](mid_ft) + Vinput

        for i in range(self.r):
            Q = self._modules['fcq_%d_%d' % (self.nb - 1, i)](Vinput)
            K = self._modules['fck_%d_%d' % (self.nb - 1, i)](Vinput)
            Attn = m(torch.matmul(Q, torch.transpose(K, 1, 2)) / torch.sqrt(torch.tensor(self.dk, dtype=torch.float)))
            V = (torch.matmul(Attn[:, :self.tokend + self.out_dim, :], xy[:, :, :]))
            V = V.permute(0, 2, 1)
            out_ft += (self._modules['fcp%d' % i](V))

        return out_ft

    # Initializing machine learning weigths 
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

# Combined model that integrates the TransformerEncoder and MS_Loss
class CombinedModel(nn.Module):
    def __init__(self, transformer, ms_loss, P):
        super(CombinedModel, self).__init__()
        # Initialize the transformer and MS_Loss models
        self.transformer = transformer
        self.ms_loss = ms_loss
        self.P = P

    def forward(self, inputs, y_inputs):
        # Extract the initial part of the input sequence
        u0 = y_inputs.permute(0, 1, 2)
        # Pass the input through the transformer encoder
        xf = self.transformer(inputs.permute(1, 0, 2))
        # Reshape the transformer output
        xf_transformer_input = xf.permute(1, 2, 0)
        # Concatenate the initial sequence and transformer output
        xf_transformer_input = torch.cat((u0, xf_transformer_input), 1)
        # Pass through MS_Loss model
        outputs = self.ms_loss(xf_transformer_input.double(), self.P).permute(0, 2, 1)
        return outputs



# Learning rate scheduler
def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

# Learning rate scheduling function
def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))


class TransformerProjector(nn.Module):
    """
    Project a 1D input sequence into a 1D output sequence via a small Transformer.

    Input:
      cond_seq: Tensor of shape (B, seq_len, input_dim)
    Output:
      cond_proj: Tensor of shape (B, seq_len, output_dim)
    """
    def __init__(self,
                 input_dim:   int = 1,
                 seq_len:     int = 17,
                 d_model:     int = 16,
                 nhead:       int = 2,
                 num_layers:  int = 2,
                 output_dim:  int = 1):
        super().__init__()
        # learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # project input_dim → d_model
        self.embedding = nn.Linear(input_dim, d_model)

        # a small Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # final head: d_model → output_dim, applied per time step
        self.fc = nn.Linear(d_model, output_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, cond_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
          cond_seq: (B, seq_len, input_dim)
        Returns:
          cond_proj: (B, seq_len, output_dim)
        """
        # 1) embed and add positional
        x = self.embedding(cond_seq)       # (B, seq_len, d_model)
        x = x + self.pos_embedding         # (B, seq_len, d_model)

        # 2) run through transformer
        x = self.transformer(x)            # (B, seq_len, d_model)

        # 3) project to output_dim, per time step
        cond_proj = self.fc(x).permute(0,2,1)             # (B, output_dim, seq_len) (400,3,850)

        # 4) pool down to a single step
        cond_proj = cond_proj.mean(dim=2, keepdim=True)  

        return cond_proj

class LearningCurveLogger:
    def __init__(self, num_parameters=None):
        self.train_losses = []
        self.test_losses = []
        self.num_parameters = num_parameters

    def update(self, train_loss, test_loss):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)

    def save(self, filename="learning_curve_GNOT_AM.png", csv_filename="learning_curve_GNOT_AM.csv"):
        # Save plot
        plt.figure()
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.title("Learning Curve")
        plt.savefig(filename)
        plt.close()

        # Save CSV
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ["Epoch", "Train Loss", "Test Loss"]
            if self.num_parameters is not None:
                headers.append("Num Parameters")
            writer.writerow(headers)

            for epoch, (train, test) in enumerate(zip(self.train_losses, self.test_losses)):
                row = [epoch, train, test]
                if self.num_parameters is not None:
                    row.append(self.num_parameters)
                writer.writerow(row)

import numpy as np
import torch
import pandas as pd

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 0
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# ── Data & Paths ────────────────────────────────────────────────────────────────
data_dir = '/home/ftk3187/github/GAMMA/GeoAgnosticASNO_research/ASNO/ASNO_codes/DED_melt_pool_thinwall_data_dict.mat'  # Path to the data file

# ── Data Loading ────────────────────────────────────────────────────────────────
reader  = MatReader(data_dir)

# raw arrays: shape (N_samples × T, …)
sol_data    = reader.read_field('temperature')
input_data  = reader.read_field('input_data')
conductivity_data = reader.read_field('conductivity')     # shape (N,) or (N,1)


# ── Simulation Dimensions ───────────────────────────────────────────────────────
GRID_SIZE           = 30        # spatial grid: 30×30
GRID_SIZE_TRAIN     = 6         # input_data spatial size
TIME_STEPS          = 731       # total time steps per sample
HISTORY_LENGTH      = 5
EFFECTIVE_STEPS     = TIME_STEPS - HISTORY_LENGTH  # 725

# ── Dataset Split ────────────────────────────────────────────────────────────────
NUM_SAMPLES_TOTAL = sol_data.size()[0]
NUM_TRAIN_SAMPLES = int(NUM_SAMPLES_TOTAL * 0.8)
NUM_TEST_SAMPLES = NUM_SAMPLES_TOTAL - NUM_TRAIN_SAMPLES
AUGMENTATIONS        = 20       # random perms per sample

# ── Batch & Augmented Sizes ─────────────────────────────────────────────────────
BATCH_SIZE_TRAIN    = 100
BATCH_SIZE_TEST     = BATCH_SIZE_TRAIN
N_TRAIN_TOTAL       = NUM_TRAIN_SAMPLES * AUGMENTATIONS
N_TEST_TOTAL        = NUM_TEST_SAMPLES  * AUGMENTATIONS

# ── Model Hyperparameters ───────────────────────────────────────────────────────
NUM_EPOCHS          = 25
LEARNING_RATE       = 3e-3
WEIGHT_DECAY        = 1e-2
LR_STEP_SIZE        = 100
LR_GAMMA            = 0.7

NUM_LAYERS          = 3         # e.g. # of layers in your neural operator
D_MODEL             = 50        # dimension of keys/queries/features
NUM_HEADS           = 1         # attention heads

# ── I/O Dimensions ──────────────────────────────────────────────────────────────
FEATURE_DIM         = D_MODEL
OUTPUT_DIM          = GRID_SIZE * GRID_SIZE  # e.g. for a 30×30 output field

# ── Index Generation ────────────────────────────────────────────────────────────
# Total timesteps used per sample for indexing
# 1) pick **sample**–level indices, not flattened
sample_train_idx = np.arange(NUM_TRAIN_SAMPLES)              # e.g. [0,1,…,40]
sample_test_idx  = np.arange(NUM_TRAIN_SAMPLES, 
                             NUM_TRAIN_SAMPLES+NUM_TEST_SAMPLES)

# 2) index by samples
sol_train = sol_data[sample_train_idx]                       # (41, 730, 30, 30)
y_train   = input_data[sample_train_idx]                     # (41, 730, features)

sol_test  = sol_data[sample_test_idx]                        # (1, 730, 30, 30)
y_test    = input_data[sample_test_idx]                      # (1, 730, features)

# reshape into (batch, tasks, timesteps, *spatial_dims)
sol_train = sol_train.view(NUM_TRAIN_SAMPLES, 1, TIME_STEPS, GRID_SIZE, GRID_SIZE)

# 1) grab the last time-step for each batch: shape (B,1,F)
last_step = y_train[:, -1:, :]    # [:, -1:, :] keeps the time‐axis

# 2) concatenate along the time dimension (dim=1)
y_train = torch.cat([y_train, last_step], dim=1)

y_train   = y_train.view(NUM_TRAIN_SAMPLES, 1, TIME_STEPS, GRID_SIZE_TRAIN)

sol_test  = sol_test.view(NUM_TEST_SAMPLES, 1, TIME_STEPS, GRID_SIZE, GRID_SIZE)

# 1) grab the last time-step for each batch: shape (B,1,F)
last_step = y_test[:, -1:, :]    # [:, -1:, :] keeps the time‐axis

# 2) concatenate along the time dimension (dim=1)
y_test = torch.cat([y_test, last_step], dim=1)

y_test    = y_test.view(NUM_TEST_SAMPLES, 1, TIME_STEPS, GRID_SIZE_TRAIN)

# Load and reshape conductivity (you already have this):
conductivity_data    = reader.read_field('conductivity')      # shape (N,) or (N,1)
CONDUCTIVITY_DATA    = conductivity_data.view(NUM_SAMPLES_TOTAL, 17)
cond_train           = CONDUCTIVITY_DATA[:NUM_TRAIN_SAMPLES, :].view(NUM_TRAIN_SAMPLES, 1, 1, 17)
cond_test            = CONDUCTIVITY_DATA[NUM_TRAIN_SAMPLES:, :].view(NUM_TEST_SAMPLES,   1, 1, 17)

# Now do exactly the same for heat_capacity:
heat_capacity_data   = reader.read_field('heat_capacity')     # shape (N,) or (N,1)
HEAT_CAPACITY_DATA   = heat_capacity_data.view(NUM_SAMPLES_TOTAL, 17)
heat_train           = HEAT_CAPACITY_DATA[:NUM_TRAIN_SAMPLES, :].view(NUM_TRAIN_SAMPLES, 1, 1, 17)
heat_test            = HEAT_CAPACITY_DATA[NUM_TRAIN_SAMPLES:, :].view(NUM_TEST_SAMPLES,   1, 1, 17)

# Now do exactly the same for material_property:
material_properties_data   = reader.read_field('material_properties')     # shape (N,) or (N,1)
MATERIAL_PROPERTIES_DATA   = material_properties_data.view(NUM_SAMPLES_TOTAL, 3)
material_properties_train           = MATERIAL_PROPERTIES_DATA[:NUM_TRAIN_SAMPLES, :].view(NUM_TRAIN_SAMPLES, 1, 1, 3)
material_properties_test           = MATERIAL_PROPERTIES_DATA[NUM_TRAIN_SAMPLES:, :].view(NUM_TEST_SAMPLES, 1, 1, 3)

import torch

def make_history_windows(sol: torch.Tensor,
                         y:   torch.Tensor,
                         history: int,
                         cond: torch.Tensor = None,
                         heat: torch.Tensor = None,
                         mat_prop: torch.Tensor = None):
    """
    Turn (B,1,T,...) sol and y into sliding windows of length history+1.
    Pass cond/heat/mat_prop through unchanged.

    Args:
      sol      (B,1,T,...)        state tensor over time
      y        (B,1,T,...)        target tensor over time
      history  how many past frames to include (gives history+1 inputs)
      cond     (B,1,1,Cc)         per-sample conductivity
      heat     (B,1,1,Ch)         per-sample heat capacity
      mat_prop (B,1,1,Cm)         per-sample material properties

    Returns:
      inputs   list of length history+1, each (B,1,T-history,...)
      targets  (B,1,T-history,...)
      cond     same tensor as passed in (or None)
      heat     same tensor as passed in (or None)
      mat_prop same tensor as passed in (or None)
    """
    B, tasks, T = sol.shape[:3]
    L = T - history
    inputs = [sol[:, :, i : i + L, ...] for i in range(history + 1)]
    targets = y[:, :, history : history + L, ...]
    return inputs, targets, cond, heat, mat_prop


def prepare_for_model(sol: torch.Tensor,
                      y:   torch.Tensor,
                      history: int,
                      S:    int,
                      S_train: int,
                      cond: torch.Tensor = None,
                      heat: torch.Tensor = None,
                      mat_prop: torch.Tensor = None):
    """
    Combines sliding-window + reshape/permute steps and returns cond/heat/mat_prop.

    Args:
      sol      (B,1,T,S,S)
      y        (B,1,T,S_train)
      history  how many past frames you want (e.g. 5 for 6 inputs)
      S        spatial grid size for sol (e.g. 30)
      S_train  spatial grid size for y (e.g. 6)
      cond     (B,1,1,Cc) per-sample conductivity
      heat     (B,1,1,Ch) per-sample heat capacity
      mat_prop (B,1,1,Cm) per-sample material properties

    Returns:
      U_list   list of length history+1; each (B, S*S,  T-history)
      Y        tensor (B, S_train, T-history)
      cond     same as input
      heat     same as input
      mat_prop same as input
    """
    inputs, targets, cond_out, heat_out, mat_prop_out = \
        make_history_windows(sol, y, history, cond, heat, mat_prop)

    B = sol.shape[0]
    L = sol.shape[2] - history

    # reshape & permute
    U_list = []
    for u in inputs:
        u = u.squeeze(1)                    # (B,L,S,S)
        u = u.reshape(B, L, S * S)          # (B,L,S*S)
        U_list.append(u.permute(0, 2, 1).float())  # (B,S*S,L)

    y_t = targets.squeeze(1).reshape(B, L, S_train)  # (B,L,S_train)
    Y   = y_t.permute(0, 2, 1).float()               # (B,S_train,L)

    cond_out = cond_out.squeeze(1).permute(0, 2, 1)
    heat_out = heat_out.squeeze(1).permute(0, 2, 1)
    mat_prop_out = mat_prop_out.squeeze(1).permute(0, 2, 1)


    return U_list, Y, cond_out, heat_out, mat_prop_out

def prepare_random_samples(sol_inputs, sol_y,
                           cond, heat, mat_prop,
                           n_randperm, n_seq, dk):
    """
    Create random & sequential feature sampling for input, target, and per-sample scalars.

    Args:
        sol_inputs (list of torch.Tensor): [u0, u1, ..., u5], each of shape (N, F, T)
        sol_y       (torch.Tensor):       target tensor shape (N, F_y, T)
        cond        (torch.Tensor):       conductivity shape (N, Cc, 1)
        heat        (torch.Tensor):       heat_capacity shape (N, Ch, 1)
        mat_prop    (torch.Tensor):       material_properties shape (N, Cm, 1)
        n_randperm  (int):                number of random permutations
        n_seq       (int):                number of sequential slices
        dk          (int):                slice width for each sample

    Returns:
        sol_y_train (torch.Tensor): ( (n_randperm+n_seq)*N, F_y, dk )
        u5_train    (torch.Tensor): ( (n_randperm+n_seq)*N,   F, dk )
        u_seq1      (torch.Tensor): ( (n_randperm+n_seq)*N, 5*F, dk )
        cond_train  (torch.Tensor): ( (n_randperm+n_seq)*N,  Cc,  1 )
        heat_train  (torch.Tensor): ( (n_randperm+n_seq)*N,  Ch,  1 )
        mat_train   (torch.Tensor): ( (n_randperm+n_seq)*N,  Cm,  1 )
    """
    N, F, T = sol_inputs[0].shape

    # prepare collectors
    u_batches    = [[] for _ in sol_inputs]
    y_batches    = []
    cond_batches = []
    heat_batches = []
    mat_batches  = []

    # 1) Random permutations
    for _ in range(n_randperm):
        idx = torch.randperm(T)[:dk]
        for i, u in enumerate(sol_inputs):
            u_batches[i].append(u[:, :, idx])
        y_batches.append(sol_y[:, :, idx])
        cond_batches.append(cond)
        heat_batches.append(heat)
        mat_batches.append(mat_prop)

    # 2) Sequential slices
    start = 0
    for _ in range(n_seq):
        idx = torch.arange(start, start + dk)
        for i, u in enumerate(sol_inputs):
            u_batches[i].append(u[:, :, idx])
        y_batches.append(sol_y[:, :, idx])
        cond_batches.append(cond)
        heat_batches.append(heat)
        mat_batches.append(mat_prop)
        start += dk

    # concat along batch dim
    u_train     = [torch.cat(lst, dim=0) for lst in u_batches]   # list of 6 tensors
    sol_y_train = torch.cat(y_batches,    dim=0)                # (M*N, F_y, dk)
    cond_train  = torch.cat(cond_batches, dim=0)                # (M*N, Cc, 1)
    heat_train  = torch.cat(heat_batches, dim=0)                # (M*N, Ch, 1)
    mat_train   = torch.cat(mat_batches,  dim=0)                # (M*N, Cm, 1)

    # normalize inputs & targets
    x_norm = GaussianNormalizer(sol_inputs[0])
    f_norm = GaussianNormalizer(sol_y_train)
    u_norm = [x_norm.encode(u) for u in u_train]
    sol_y_norm = f_norm.encode(sol_y_train)
    cond_train_norm = GaussianNormalizer(cond_train)
    cond_train = cond_train_norm.encode(cond_train)
    heat_train_norm = GaussianNormalizer(heat_train)
    heat_train = heat_train_norm.encode(heat_train)
    mat_train_norm = GaussianNormalizer(mat_train)
    mat_train = mat_train_norm.encode(mat_train)

    # extract u5 and stack u0..u4
    u5_train = u_norm[5]
    u_seq1   = torch.cat(u_norm[:5], dim=1)

    return sol_y_norm, u5_train, u_seq1, cond_train, heat_train, mat_train, x_norm, f_norm



# ── Example usage ─────────────────────────────────────────────────────────────
# sol_train: (N_train,1,T,S,S)
# y_train  : (N_train,1,T,S_train)
# cond_train: (N_train,1,1,Cc), heat_train: (N_train,1,1,Ch), material_properties_train: (N_train,1,1,Cm)
H = 5
U_list, Y, c, h, m = prepare_for_model(
    sol_train, y_train,
    history=H,
    S=GRID_SIZE,
    S_train=GRID_SIZE_TRAIN,
    cond=cond_train,
    heat=heat_train,
    mat_prop=material_properties_train
)

dk = 50 # slice width

# Example call:
sol_y_train, u5_train, u_seq1, cond_train, heat_train, mat_train, x_norm, f_norm =    prepare_random_samples(U_list, Y, c, h, m,n_randperm=2, n_seq=2, dk=dk)

# Print the shapes of the tensors
print("Shape of sol_y_train:", sol_y_train.shape)
print("Shape of u5_train:", u5_train.shape)
print("Shape of u_seq1:", u_seq1.shape)
print("Shape of cond_train:", cond_train.shape)
print("Shape of heat_train:", heat_train.shape)
print("Shape of mat_train:", mat_train.shape)

## Test Data Generation
H = 5
U_list_test, Y_test, c_test, h_test, m_test = prepare_for_model(
    sol_test, y_test,
    history=H,
    S=GRID_SIZE,
    S_train=GRID_SIZE_TRAIN,
    cond=cond_test,
    heat=heat_test,
    mat_prop=material_properties_test
)

# Example call:
sol_y_test, u5_test, u_seq1_test, cond_test, heat_test, mat_test, x_norm, f_norm =    prepare_random_samples(U_list_test, Y_test, c_test, h_test, m_test,n_randperm=2, n_seq=2, dk=dk)
# Print the shapes of the tensors
print("Shape of sol_y_test:", sol_y_test.shape)
print("Shape of u5_test:", u5_test.shape)
print("Shape of u_seq1_test:", u_seq1_test.shape)
print("Shape of cond_test:", cond_test.shape)
print("Shape of heat_test:", heat_test.shape)
print("Shape of mat_test:", mat_test.shape)

BATCH_SIZE_TRAIN = 50
BATCH_SIZE_TEST  = 10

# DataLoader for training data
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u_seq1, u5_train, sol_y_train, cond_train, heat_train, mat_train),
                                           batch_size=BATCH_SIZE_TRAIN, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u_seq1_test, u5_test, sol_y_test, cond_test, heat_test, mat_test),
                                              batch_size=BATCH_SIZE_TEST, shuffle=False)

heat_transformer_output_size = 8
cond_transformer_output_size = 8

# Input sequence reshaping
#u_seq = u_seq1.permute(0, 2, 1).float()  # Reshape to desired dimensions, # ntrain*n_randperm X sall X 15
#u5_train = u5_train.permute(0, 2, 1)  # Ensure u5_train has the correct dimensions, # ntrain*n_randperm X sall X 3
tokend = u5_train.shape[1]+sol_y_train.shape[1]+mat_test.shape[1]+heat_transformer_output_size+cond_transformer_output_size  # This is an add-on to the neural operator, =S=3

# Example usage for prediction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model and training parameter setup
embed_dim = 10  # Embedding dimension for transformer
num_heads = 5  # Number of attention heads
num_layers = 2  # Number of transformer layers

# Define input and output dimensions based on the concatenated test sequence
input_dim = u_seq1.size()[1]
output_dim = u5_train.size()[1]
# Before you enter the ablation loop, prepare a list to collect results:
results = []

# iterate ablation configs
for cfg in ablation_configs:
    odir = cfg['output_dir']
    learning_rate, wd, gamma = cfg['learning_rate'], cfg['weight_decay'], cfg['gamma']

    transformer_encoder = TransformerEncoder(input_dim, output_dim, embed_dim, num_heads, num_layers).to(device).float()
    P = torch.zeros((1, output_dim, output_dim)).to(device)

    # Model parameters
    nb = 4  # Number of layers in the model
    dk = dk  # Dimension of the key/query vectors
    r=1

    NAO_featured = dk
    NAO_output_dim = GRID_SIZE**2

    model = MS_Loss(tokend, r, dk, nb, NAO_featured, NAO_output_dim).to(device)

    model_cond = TransformerProjector(input_dim=1, seq_len=17, d_model=2, nhead=1, num_layers=2, output_dim=cond_transformer_output_size).to(device).double()
    model_heat = TransformerProjector(input_dim=1, seq_len=17, d_model=2, nhead=1, num_layers=2, output_dim=heat_transformer_output_size).to(device).double()

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #combined_model.to(device)  # Move model to GPU if available
    #optimizer = torch.optim.Adam(list(model.parameters())+list(transformer_encoder.parameters()) + list(model_cond.parameters())+list(model_heat.parameters()), lr=learning_rate, weight_decay=wd)  # Optimizer


    # Total number of trainable parameters
    total_params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)     
    total_params_transformer_encoder = sum(p.numel() for p in transformer_encoder.parameters() if p.requires_grad)     
    total_params_model_cond = sum(p.numel() for p in model_cond.parameters() if p.requires_grad)     
    total_params_model_heat = sum(p.numel() for p in model_heat.parameters() if p.requires_grad)     
    total_params = total_params_model + total_params_transformer_encoder + total_params_model_cond + total_params_model_heat

    print(f"Total number of trainable parameters: {total_params}")

    # Initialize the logger
    logger = LearningCurveLogger(num_parameters=total_params)

    optimizer = torch.optim.Adam(list(model.parameters())+list(transformer_encoder.parameters()) + list(model_cond.parameters())+list(model_heat.parameters()), lr=learning_rate, weight_decay=wd)  # Optimizer
    criterion = nn.MSELoss()  # Loss function
    myloss = LpLoss(size_average=False)  # Custom loss function

    # Initialize variables to track the best loss and epoch
    train_loss_best = train_loss_lowest = test_loss_best = 1e8
    best_epoch = 0

    num_epochs = 200
    x_normalizer = x_norm 
    step_size = 100  # Step size for learning rate scheduler
    # Total number of training and test samples
    ntrain_total = N_TRAIN_TOTAL * 2
    ntest_total = N_TEST_TOTAL * 2

    # Loop over epochs
    for epoch in range(num_epochs):
        optimizer = scheduler(optimizer, LR_schedule(learning_rate, epoch, step_size, gamma))  # Adjust learning rate
        #combined_model.train()  # Set model to training mode
        transformer_encoder.train()
        model.train()
        model_cond.train()
        model_heat.train()
        epoch_loss = 0
        t1 = default_timer()  # Start timer
        train_l2 = 0

        # Loop over training batches
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, targets, y_inputs, cond_batch, heat_batch, mat_batch = batch
            inputs, targets, y_inputs = inputs.to(device), targets.to(device), y_inputs.to(device)  # Move data to GPU
            this_batch_size = inputs.shape[0]

            optimizer.zero_grad()  # Clear gradients

            # Forward pass through the combined model
            # inputs: batch X 15 X sall
            #outputs = combined_model(inputs, y_inputs)
            # Inside the training loop or during forward pass
            input0 = inputs.permute(0, 2, 1).to(device)
            input1 = transformer_encoder(input0)

            model_cond_out = model_cond(cond_batch.to(device).double())
            model_heat_out = model_heat(heat_batch.to(device).double())
            model_material_out = mat_batch.to(device).double()

            # assume input1 has shape (400, 900, 50)
            T = input1.size(1)   # 50

            # model_cond_out, model_heat_out, model_material_out each have shape (400,1,1)
            # replicate across the time axis:
            model_cond_out = model_cond_out.repeat(1, 1, T)      # (400,1,50)
            model_heat_out = model_heat_out.repeat(1, 1, T)      # (400,1,50)
            model_material_out = model_material_out.repeat(1, 1, T)# (400,1,50)

            combined_input = torch.cat((y_inputs.to(device), input1.permute(0, 2, 1), model_cond_out, model_heat_out, model_material_out), 1)

            outputs = model(combined_input.double()).permute(0, 2, 1).to(device)
            outputs = x_normalizer.decode(outputs)  # Decode the outputs

            # Compute the loss
            loss = myloss(outputs.reshape(this_batch_size, -1), x_normalizer.decode(targets).reshape(this_batch_size, -1))

            epoch_loss += loss.item()
            loss.backward()  # Backpropagation

            optimizer.step()  # Update model parameters
            train_l2 += loss.item()

        train_l2 /= ntrain_total  # Normalize the training loss

        # Track the lowest training loss
        if train_l2 < train_loss_lowest:
            train_loss_lowest = train_l2

        # Save the model with the best performance on the validation set
        if train_l2 < train_loss_best:
            transformer_encoder.train()
            model.train()
            model_cond.train()
            model_heat.train()
            test_l2 = 0.0

            # Evaluate on the test set
            with torch.no_grad():
                for batch in test_dataloader:
                    inputs, targets, y_inputs, cond_batch, heat_batch, mat_batch = batch
                    inputs, targets, y_inputs = inputs.to(device), targets.to(device), y_inputs.to(device)  # Move data to GPU
                    this_batch_size = inputs.shape[0]

                    optimizer.zero_grad()  # Clear gradients

                    # Forward pass through the combined model
                    # inputs: batch X 15 X sall
                    #outputs = combined_model(inputs, y_inputs)
                    # Inside the training loop or during forward pass
                    input0 = inputs.permute(0, 2, 1).to(device)
                    input1 = transformer_encoder(input0)

                    model_cond_out = model_cond(cond_batch.to(device).double())
                    model_heat_out = model_heat(heat_batch.to(device).double())
                    model_material_out = mat_batch.to(device).double()

                    # assume input1 has shape (400, 900, 50)
                    T = input1.size(1)   # 50

                    # model_cond_out, model_heat_out, model_material_out each have shape (400,1,1)
                    # replicate across the time axis:
                    model_cond_out = model_cond_out.repeat(1, 1, T)      # (400,1,50)
                    model_heat_out = model_heat_out.repeat(1, 1, T)      # (400,1,50)
                    model_material_out = model_material_out.repeat(1, 1, T)# (400,1,50)

                    combined_input = torch.cat((y_inputs.to(device), input1.permute(0, 2, 1), model_cond_out, model_heat_out, model_material_out), 1)

                    outputs = model(combined_input.double()).permute(0, 2, 1).to(device)
                    outputs = x_normalizer.decode(outputs)  # Decode the outputs
                    # Compute test loss
                    test_l2 += myloss(outputs.reshape(this_batch_size, -1).to(device), x_normalizer.decode(targets).reshape(this_batch_size, -1).to(device)).item()

            test_l2 /= ntest_total  # Normalize the test loss

            logger.update(train_l2, test_l2)

            # If the current test loss is the best, save the model
            if test_l2 < test_loss_best:
                best_epoch = epoch
                train_loss_best = train_l2
                test_loss_best = test_l2
                torch.save(model.state_dict(), odir + "/NAO_DT_mean_mt")  # Save model state
                torch.save(transformer_encoder.state_dict(), odir + "/transformer_encoder_DT_mean_mt")  # Save model state
                torch.save(model_cond.state_dict(), odir + "/model_cond_DT")  # Save model state
                torch.save(model_heat.state_dict(), odir + "/model_heat_DT")  # Save model state

                t2 = default_timer()  # End timer
                print(f'>> depth{nb}, epoch [{(epoch + 1)}], '
                    f'runtime: {(t2 - t1):.2f}s, train err: {train_l2:.5f}, test err: {test_l2:.5f}'
                    f'learning rate: {learning_rate}, weight decay: {wd}, gamma: {gamma}, ')
            else:
                t2 = default_timer()
                print(f'>> depth{nb}, epoch [{(epoch + 1)}], '
                    f'runtime: {(t2 - t1):.2f}s, train err: {train_l2:.5f}, test err: {test_l2:.5f}, (best: [{best_epoch}], '
                    f'{train_loss_best:.5f}/{test_loss_best:.5f})'
                    f'learning rate: {learning_rate}, weight decay: {wd}, gamma: {gamma}, ')
        else:
            t2 = default_timer()
            print(f'>> depth{nb}, epoch [{(epoch + 1)}], '
                f'runtime: {(t2 - t1):.2f}s, train err: {train_l2:.5f} (best: [{best_epoch}], '
                f'{train_loss_best:.5f}/{test_loss_best:.5f})'
                f'learning rate: {learning_rate}, weight decay: {wd}, gamma: {gamma}, ')

        logger.save(filename=odir+"/learning_curve_GNOT_AM.png", csv_filename=odir+"/learning_curve_GNOT_AM.csv")
        results.append({
            'learning_rate': learning_rate,
            'weight_decay': wd,
            'gamma': gamma,
            'best_test_loss': test_loss_best
        })
        df = pd.DataFrame(results)
        df.to_csv(base_output_dir + "/ablation_results.csv", index=False)




# CUDA_VISIBLE_DEVICES=1 python /home/vnk3019/foundational_model_NAO/DED_example/DED_material/training_model.pyvscode-remote://ssh-remote%2B7b22686f73744e616d65223a224c656961227d/home/vnk3019/foundational_model_NAO/DED_example/DED_material/DED_melt_pool_thinwall_data_dict.mat