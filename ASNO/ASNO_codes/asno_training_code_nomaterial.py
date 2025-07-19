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

# Filename for saving the model
model_filename = "Transformer-NAO"

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


class LearningCurveLogger:
    def __init__(self, num_parameters=None):
        self.train_losses = []
        self.test_losses = []
        self.num_parameters = num_parameters

    def update(self, train_loss, test_loss):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)

    def save(self, filename="learning_curve_ASNO_AM.png", csv_filename="learning_curve_ASNO_AM.csv"):
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

def main(learning_rate, gamma, wd, nb, dk, save_filename, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    sall = dk  # Feature dimension (same as dk here)
    num_epochs = 1000  # Number of training epochs

    # Random seed for reproducibility
    seed = 0
    torch.manual_seed(seed)  # Set seed for torch (CPU)
    torch.cuda.manual_seed(seed)  # Set seed for torch (GPU, if available)
    np.random.seed(seed)  # Set seed for numpy

    # Other hyperparameters
    r = 1  # Number of attention heads (usually in attention models)
    trial = 1  # Trial ID (could be useful for multiple runs)
    step_size = 100  # Step size for learning rate scheduler
    epochs = 25  # Number of training epochs
    S = 30  # Spatial resolution (e.g., 21x21 grid)
    dt = 5e-5  # Time step size
    df = dk  # Feature dimension (same as dk here)
    sall = dk  # Feature dimension (same as dk here)

    # Data configuration
    data_dir = '/home/vnk3019/DED_melt_pool_thinwall_data_dict.mat'  # Path to the data file

    # Dataset splitting
    ntotal = 41  # Total number of samples
    ntrain = 41  # Number of training samples
    ntest = 1  # Number of test samples

    # Time series configuration
    n_timesteps = 730  # Total number of time steps
    nt = n_timesteps - 5  # Effective time steps after accounting for history
    n_randperm = 20  # Number of random permutations (data augmentation)

    # Batch sizes
    batch_size_train = 100  # Training batch size
    batch_size_test = batch_size_train  # Test batch size (same as training batch size)

    # Total number of training and test samples
    ntrain_total = ntrain * n_randperm
    ntest_total = ntest * n_randperm

    # Input/output dimensions
    featured = sall  # Input dimension (feature dimension)
    out_dim = S**2  # Output dimension (e.g., 21x21 grid)


    # Sampling configurations
    sample_per_task = 1  # Number of samples per task
    rands = 100  # Number of random samples to consider (maybe a setting for randomness)

    # Splitting indices for training and testing
    train_indexes = np.arange(0, 29971)  # Indices for training data
    test_indexes = np.arange(0, 29971)  # Indices for testing data

# Generate lists for training, validation, and testing
    train_list = list(train_indexes)[:ntrain *n_timesteps * sample_per_task]
    valid_list = list(train_indexes)[-ntest *n_timesteps * sample_per_task:]  # Validation data (can be adjusted as needed)
    test_list = list(train_indexes)[-ntest *n_timesteps * sample_per_task:]

    # ── Data & Paths ────────────────────────────────────────────────────────────────
    data_dir = '/home/ftk3187/github/GAMMA/GeoAgnosticASNO_research/ASNO/ASNO_codes/DED_melt_pool_thinwall_data_dict.mat'  # Path to the data file


    # Load the dataset using MatReader
    reader = MatReader(data_dir)

    S_train = 6

    # Extract fields from the dataset for training and testing
    sol_train = reader.read_field('temperature')[train_list, :].view(ntrain, sample_per_task, n_timesteps, S, S)
    y_train = reader.read_field('input_data')[train_list, :].view(ntrain, sample_per_task, n_timesteps, S_train)

    sol_test = reader.read_field('temperature')[train_list, :].view(ntrain, sample_per_task, n_timesteps, S, S)
    y_test = reader.read_field('input_data')[train_list, :].view(ntrain, sample_per_task, n_timesteps, S_train)

    # Prepare training data by extracting consecutive time steps (history)
    sol_train_u0 = sol_train[:, :, :n_timesteps - 5, ...]
    sol_train_u1 = sol_train[:, :, 1:n_timesteps - 4, ...]
    sol_train_u2 = sol_train[:, :, 2:n_timesteps - 3, ...]
    sol_train_u3 = sol_train[:, :, 3:n_timesteps - 2, ...]
    sol_train_u4 = sol_train[:, :, 4:n_timesteps - 1, ...]
    sol_train_u5 = sol_train[:, :, 5:n_timesteps, ...]
    sol_train_y = y_train[:, :, 5:n_timesteps, ...]

    # Reshape and permute tensors to match the required input format
    sol_train_u0 = sol_train_u0[:ntrain, ...].reshape(ntrain, n_timesteps-5, S ** 2).permute(0, 2, 1).float()
    sol_train_u1 = sol_train_u1[:ntrain, ...].reshape(ntrain, n_timesteps-5, S ** 2).permute(0, 2, 1).float()
    sol_train_u2 = sol_train_u2[:ntrain, ...].reshape(ntrain, n_timesteps-5, S ** 2).permute(0, 2, 1).float()
    sol_train_u3 = sol_train_u3[:ntrain, ...].reshape(ntrain, n_timesteps-5, S ** 2).permute(0, 2, 1).float()
    sol_train_u4 = sol_train_u4[:ntrain, ...].reshape(ntrain, n_timesteps-5, S ** 2).permute(0, 2, 1).float()
    sol_train_u5 = sol_train_u5[:ntrain, ...].reshape(ntrain, n_timesteps-5, S ** 2).permute(0, 2, 1).float()
    sol_train_y = sol_train_y[:ntrain, ...].reshape(ntrain, n_timesteps-5, S_train).permute(0, 2, 1).float()

    # Initialize lists to store randomly selected data for each time step
    u0_df = []
    u1_df = []
    u2_df = []
    u3_df = []
    u4_df = []
    u5_df = []
    y_df = []

    # Number of features to randomly select
    n_f = n_timesteps - 5

    # Randomly select features and append them to respective lists
    for _ in range(n_randperm):
        crand = torch.randperm(n_f)[:df]
        u0_df.append(sol_train_u0[..., crand])
        u1_df.append(sol_train_u1[..., crand])
        u2_df.append(sol_train_u2[..., crand])
        u3_df.append(sol_train_u3[..., crand])
        u4_df.append(sol_train_u4[..., crand])
        u5_df.append(sol_train_u5[..., crand])
        y_df.append(sol_train_y[..., crand])


    # Concatenate the selected features into the final training datasets
    u0_train = torch.cat(u0_df, dim=0)
    u1_train = torch.cat(u1_df, dim=0)
    u2_train = torch.cat(u2_df, dim=0)
    u3_train = torch.cat(u3_df, dim=0)
    u4_train = torch.cat(u4_df, dim=0)
    u5_train = torch.cat(u5_df, dim=0)
    sol_y_train = torch.cat(y_df, dim=0)

    # Normalize the training datasets using a Gaussian normalizer
    x_normalizer = GaussianNormalizer(sol_train)
    f_normalizer = GaussianNormalizer(sol_y_train)

    u0_train = x_normalizer.encode(u0_train)
    u1_train = x_normalizer.encode(u1_train)
    u2_train = x_normalizer.encode(u2_train)
    u3_train = x_normalizer.encode(u3_train)
    u4_train = x_normalizer.encode(u4_train)
    sol_y_train = f_normalizer.encode(sol_y_train)
    u5_train = x_normalizer.encode(u5_train)  ## Not encoding u5_train

    u_seq1 = torch.cat((u0_train, u1_train, u2_train, u3_train, u4_train), dim=1) # ntrain*n_randperm X 15 X sall

    # Define input and output dimensions based on the concatenated sequence
    input_dim = u_seq1.size()[1]
    output_dim = u5_train.size()[1]

    # Prepare testing data by extracting consecutive time steps (history)
    sol_test_u0 = sol_test[:, :, :n_timesteps - 5, ...]
    sol_test_u1 = sol_test[:, :, 1:n_timesteps - 4, ...]
    sol_test_u2 = sol_test[:, :, 2:n_timesteps - 3, ...]
    sol_test_u3 = sol_test[:, :, 3:n_timesteps - 2, ...]
    sol_test_u4 = sol_test[:, :, 4:n_timesteps - 1, ...]
    sol_test_u5 = sol_test[:, :, 5:n_timesteps, ...]
    sol_test_y = y_test[:, :, 5:n_timesteps, ...]

    # Reshape and permute the test tensors to match the required input format
    sol_test_u0 = sol_test_u0[:ntest, ...].reshape(ntest, n_timesteps-5, S ** 2).permute(0, 2, 1).float()
    sol_test_u1 = sol_test_u1[:ntest, ...].reshape(ntest, n_timesteps-5, S ** 2).permute(0, 2, 1).float()
    sol_test_u2 = sol_test_u2[:ntest, ...].reshape(ntest, n_timesteps-5, S ** 2).permute(0, 2, 1).float()
    sol_test_u3 = sol_test_u3[:ntest, ...].reshape(ntest, n_timesteps-5, S ** 2).permute(0, 2, 1).float()
    sol_test_u4 = sol_test_u4[:ntest, ...].reshape(ntest, n_timesteps-5, S ** 2).permute(0, 2, 1).float()
    sol_test_u5 = sol_test_u5[:ntest, ...].reshape(ntest, n_timesteps-5, S ** 2).permute(0, 2, 1).float()
    sol_test_y = sol_test_y[:ntest, ...].reshape(ntest, n_timesteps-5, S_train).permute(0, 2, 1).float()

    # Initialize lists to store randomly selected data for each time step
    u0_df = []
    u1_df = []
    u2_df = []
    u3_df = []
    u4_df = []
    u5_df = []
    y_df = []

    for _ in range(n_randperm):
        crand = torch.randperm(n_f)[:sall]
        u0_df.append(sol_test_u0[..., crand])
        u1_df.append(sol_test_u1[..., crand])
        u2_df.append(sol_test_u2[..., crand])
        u3_df.append(sol_test_u3[..., crand])
        u4_df.append(sol_test_u4[..., crand])
        u5_df.append(sol_test_u5[..., crand])
        y_df.append(sol_test_y[..., crand])


    # Concatenate the selected features into the final training datasets
    sol_test_u0 = torch.cat(u0_df, dim=0) # ntrain*n_randperm X 3 X sall
    sol_test_u1 = torch.cat(u1_df, dim=0)
    sol_test_u2 = torch.cat(u2_df, dim=0)
    sol_test_u3 = torch.cat(u3_df, dim=0)
    sol_test_u4 = torch.cat(u4_df, dim=0)
    sol_test_u5 = torch.cat(u5_df, dim=0)
    sol_test_y = torch.cat(y_df, dim=0)


    # Normalize the test tensors using the same normalizer used for training data
    #YY: should use the same normalizer for train and test
    #x_normalizer = GaussianNormalizer(sol_prev_test)  # Assuming it is trained on your training dataset

    u_test_u0 = x_normalizer.encode(sol_test_u0)
    u_test_u1 = x_normalizer.encode(sol_test_u1)
    u_test_u2 = x_normalizer.encode(sol_test_u2)
    u_test_u3 = x_normalizer.encode(sol_test_u3)
    u_test_u4 = x_normalizer.encode(sol_test_u4)
    y_test = f_normalizer.encode(sol_test_y)
    u_test_u5 = x_normalizer.encode(sol_test_u5)

    # Concatenate normalized test tensors incrementally to form the test sequence
    #u_seq_test = u_test_u0
    #for u_test in [u_test_u1, u_test_u2, u_test_u3, u_test_u4]:
    #    u_seq_test = torch.cat((u_seq_test, u_test), dim=1) # ntest X 15 X sall
    u_seq_test = torch.cat((u_test_u0, u_test_u1, u_test_u2, u_test_u3, u_test_u4), dim=1) # ntest X 15 X sall

    # Define input and output dimensions based on the concatenated test sequence
    input_dim = u_seq_test.size()[1]
    output_dim = sol_test_u5.size()[1]


    # Example usage for prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model and training parameter setup
    embed_dim = 100  # Embedding dimension for transformer
    num_heads = 50  # Number of attention heads
    num_layers = 3  # Number of transformer layers

    # Input sequence reshaping
    #u_seq = u_seq1.permute(0, 2, 1).float()  # Reshape to desired dimensions, # ntrain*n_randperm X sall X 15
    #u5_train = u5_train.permute(0, 2, 1)  # Ensure u5_train has the correct dimensions, # ntrain*n_randperm X sall X 3
    tokend = u0_train.shape[1]+6  # This is an add-on to the neural operator, =S=3

    # Initialize transformer encoder and MS_Loss models
    # input_dim: 15, output_dim: 3, embed_dim: 2, num_heads: 1, num_layers: 3
    transformer_encoder = TransformerEncoder(input_dim, output_dim, embed_dim, num_heads, num_layers).to(device).float()
    P = torch.zeros((1, output_dim, output_dim)).to(device)

    NAO_featured = sall
    NAO_output_dim = S**2

    model = MS_Loss(tokend, r, dk, nb, NAO_featured, NAO_output_dim).to(device)
    #combined_model = CombinedModel(transformer_encoder, model, P).to(device)

    # DataLoader for training data
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u_seq1, u5_train, sol_y_train),
                                            batch_size=batch_size_train, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u_seq_test, u_test_u5, y_test),
                                                batch_size=batch_size_test, shuffle=False)

    # Print input/output dimensions and the number of previous time steps used
    print("Input dimension:", input_dim, "Output dimension:", output_dim, "Number of previous timesteps:", int(input_dim / output_dim))

    learning_rate = 3e-4  # Initial learning rate

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #combined_model.to(device)  # Move model to GPU if available
    optimizer = torch.optim.Adam(list(model.parameters())+list(transformer_encoder.parameters()), lr=learning_rate, weight_decay=wd)  # Optimizer
    criterion = nn.MSELoss()  # Loss function
    myloss = LpLoss(size_average=False)  # Custom loss function

    # Initialize variables to track the best loss and epoch
    train_loss_best = train_loss_lowest = test_loss_best = 1e8
    best_epoch = 0

    # Total number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) + \
                sum(p.numel() for p in transformer_encoder.parameters() if p.requires_grad)
    
    logger = LearningCurveLogger(num_parameters=total_params)

    # Loop over epochs
    for epoch in range(num_epochs):
        optimizer = scheduler(optimizer, LR_schedule(learning_rate, epoch, step_size, gamma))  # Adjust learning rate
        #combined_model.train()  # Set model to training mode
        transformer_encoder.train()
        model.train()
        epoch_loss = 0
        t1 = default_timer()  # Start timer
        train_l2 = 0

        # Loop over training batches
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, targets, y_inputs = batch
            inputs, targets, y_inputs = inputs.to(device), targets.to(device), y_inputs.to(device)  # Move data to GPU
            this_batch_size = inputs.shape[0]

            optimizer.zero_grad()  # Clear gradients

            # Forward pass through the combined model
            # inputs: batch X 15 X sall
            #outputs = combined_model(inputs, y_inputs)
            # Inside the training loop or during forward pass
            input0 = inputs.permute(0, 2, 1).to(device)
            input1 = transformer_encoder(input0)
            input2 = torch.cat((y_inputs.to(device), input1.permute(0, 2, 1)), 1)
            outputs = model(input2.double()).permute(0, 2, 1).to(device)
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
            model.eval()  # Set model to evaluation mode
            test_l2 = 0.0

            # Evaluate on the test set
            with torch.no_grad():
                for batch in test_dataloader:
                    inputs, targets, y_inputs = batch
                    #inputs, targets, y_inputs = inputs.to(device), targets.to(device)[:, :, :50], y_inputs.to(device)[:, :, :50]  # Move data to GPU
                    this_batch_size = inputs.shape[0]

                    # Forward pass through the combined model
                    #outputs = combined_model(inputs, y_inputs)
                    input0 = inputs.permute(0, 2, 1).to(device)
                    input1 = transformer_encoder(input0)
                    input2 = torch.cat((y_inputs.to(device), input1.permute(0, 2, 1)), 1)
                    outputs = model(input2.double()).permute(0, 2, 1).to(device)
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
                torch.save(model.state_dict(), "AM_TE_NAO")  # Save model state
                torch.save(transformer_encoder.state_dict(), "AM_TE_Transformer_Encoder")  # Save model state

                t2 = default_timer()  # End timer
                print(f'>> depth{nb}, epoch [{(epoch + 1)}], '
                    f'runtime: {(t2 - t1):.2f}s, train err: {train_l2:.5f}, test err: {test_l2:.5f}')
            else:
                t2 = default_timer()
                print(f'>> depth{nb}, epoch [{(epoch + 1)}], '
                    f'runtime: {(t2 - t1):.2f}s, train err: {train_l2:.5f}, test err: {test_l2:.5f}, (best: [{best_epoch}], '
                    f'{train_loss_best:.5f}/{test_loss_best:.5f})')
        else:
            t2 = default_timer()
            print(f'>> depth{nb}, epoch [{(epoch + 1)}], '
                f'runtime: {(t2 - t1):.2f}s, train err: {train_l2:.5f} (best: [{best_epoch}], '
                f'{train_loss_best:.5f}/{test_loss_best:.5f})')

    logger.save()
    f = open(save_filename, "a")
    f.write(
        f'{ntrain_total}, {learning_rate}, {gamma}, {wd}, {train_loss_lowest}, {train_loss_best}, {test_loss_best}, {best_epoch}\n')
    f.close()


if __name__ == '__main__':
    # best hyperparameters: lr = 1e-2, gammas = 0.5, wd = 1e-2
    # 4 * 3 * 9 = 12 * 9
    #lrs = [3e-3, 1e-3, 1e-2, 3e-2]
    #gammas = [0.5, 0.7, 0.9]
    #wds = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]

    lrs = [3e-3]
    gammas = [0.5]
    wds = [1e-2]
    
    # lrs = [lrs[int(sys.argv[1])]]
    # gammas = [gammas[int(sys.argv[2])]]

    seed = 0

    if len(sys.argv) > 1:
        nb = int(sys.argv[1])
        dk = int(sys.argv[2])
    else:
        nb = 2
        dk = 20
        print('-' * 100)
        print(f'>> Warning: input argument not defined. Default (nb={nb}, dk={dk}) is used!')
        print('-' * 100)

    res_dir = './res_darcy_dynamic_epochs1000'
    os.makedirs(res_dir, exist_ok=True)

    save_filename = "%s/n_nlayer%d_dk%d_setting3.txt" % (res_dir, nb, dk)
    if len(sys.argv) > 1:
        if int(sys.argv[1]) == 0 and int(sys.argv[2]) == 0:
            f = open(save_filename, "w")
            f.write(f'ntrain, learning_rate, scheduler_gamma, wd, '
                    f'overfit_train_loss, best_train_loss, best_test_loss, best_epoch\n')
            f.close()
    else:
        f = open(save_filename, "w")
        f.write(f'ntrain, learning_rate, scheduler_gamma, wd, '
                f'overfit_train_loss, best_train_loss, best_test_loss, best_epoch\n')
        f.close()

    icount = 0
    case_total = len(lrs) * len(gammas) * len(wds)
    for lr in lrs:
        for gamma in gammas:
            for wd in wds:
                icount += 1
                print("-" * 100)
                print(f'>> running case {icount}/{case_total}: lr={lr}, gamma={gamma}, wd={wd}')
                print("-" * 100)
                main(lr, gamma, wd, nb, dk, save_filename, seed)
