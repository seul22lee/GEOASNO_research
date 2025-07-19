import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
# import torch.fft
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.io import savemat

torch.set_default_tensor_type(torch.DoubleTensor)
class VAE_model(nn.Module):
    def __init__(self, Encoder, Decoder, device = 'cpu'):
        super(VAE_model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.device = device

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x, y):
        mean, log_var = self.Encoder(x, y)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z, x)

        return x_hat, mean, log_var


class Encoder(nn.Module):

    def __init__(self, input_dim=330, hidden_dim=100,  output_dim = 330):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_input3 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, output_dim)
        self.FC_var = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x, y):
        # X: batch * 880 * 330 -> Z: batch * 880 * 1
        # Z * x: batch* 880*1 * (batch) 880 * 330  -> Y: 330 * 1
        # h_ = self.LeakyReLU(self.FC_input(x) + self.LeakyReLU(self.FC_input2(y)))
        h_ = self.LeakyReLU(self.FC_input(x + y)) # 880 *300
        h_ = self.LeakyReLU(self.FC_input2(h_))
        h_ = self.LeakyReLU(self.FC_input3(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        # self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        # self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        # self.FC_output = nn.Linear(hidden_dim, output_dim)

        # self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, z, x):
        # h = self.LeakyReLU(self.FC_hidden(x))
        # h = self.LeakyReLU(self.FC_hidden2(h))
        #
        # x_hat = torch.sigmoid(self.FC_output(h))
        x_hat = torch.matmul(torch.transpose(z,1,2),x)
        return x_hat
