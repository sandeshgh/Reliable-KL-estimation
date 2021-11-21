import numpy as np
import torch
import torch.nn as nn
from spectral import SpectralNorm as spectral_norm


def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, cubic=None):
    """Generate samples from a correlated Gaussian distribution."""
    x, eps = torch.chunk(torch.randn(batch_size, 2 * dim), 2, dim=1)
    y = rho * x + torch.sqrt(torch.tensor(1. - rho**2).float()) * eps

    if cubic is not None:
        y = y ** 3

    return x, y


def rho_to_mi(dim, rho):
    """Obtain the ground truth mutual information from rho."""
    return -0.5 * np.log(1 - rho**2) * dim


def mi_to_rho(dim, mi):
    """Obtain the rho for Gaussian give ground truth mutual information."""
    return np.sqrt(1 - np.exp(-2.0 / dim * mi))


def mi_schedule(n_iter):
    """Generate schedule for increasing correlation over time."""
    mis = np.round(np.linspace(0.5, 5.5 - 1e-9, n_iter)) * 2.0
    return mis.astype(np.float32)


def mlp(dim, hidden_dim, output_dim, layers, activation):
    """Create a mlp from the configurations."""
    activation = {
        'relu': nn.ReLU
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)

def gaussian_perceptron(dim, hidden_dim, output_dim, layers, activation, lip =5):
    """Similar to mlp. Last layer is removed so output is D dim feature."""
    activation = {
        'relu': nn.ReLU
    }[activation]

    seq = [spectral_norm( nn.Linear(dim, hidden_dim), k=lip), activation()]
    for _ in range(layers):
        seq += [spectral_norm( nn.Linear(hidden_dim, hidden_dim), k=lip), activation()]
        # seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    feature_extractor = nn.Sequential(*seq)

    return feature_extractor

def gaussian_perceptron_with_op(dim, hidden_dim, output_dim, layers, activation, lip=5):
    """Create a mlp from the configurations."""
    activation = {
        'relu': nn.ReLU
    }[activation]

    seq = [spectral_norm( nn.Linear(dim, hidden_dim), k=lip), activation()]
    for _ in range(layers):
        seq += [spectral_norm( nn.Linear(hidden_dim, hidden_dim), k=lip), activation()]
        # seq += [nn.Linear(hidden_dim, hidden_dim), activation()]

    seq += [spectral_norm(nn.Linear(hidden_dim, output_dim), k=lip)]

    feature_extractor = nn.Sequential(*seq)

    return feature_extractor


class SeparableCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
        super(SeparableCritic, self).__init__()
        self._g = mlp(dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(dim, hidden_dim, embed_dim, layers, activation)

    def forward(self, x, y):
        scores = torch.matmul(self._h(y), self._g(x).t())
        return scores

class SeparableLipRKHS(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, lip , **extra_kwargs):
        super(SeparableLipRKHS, self).__init__()
        self._g = gaussian_perceptron_with_op(dim, hidden_dim, embed_dim, layers, activation, lip)
        self._h = gaussian_perceptron_with_op(dim, hidden_dim, embed_dim, layers, activation, lip)

    def forward(self, x, y):
        scores = torch.matmul(self._h(y), self._g(x).t())
        return scores

class ConcatCritic(nn.Module):
    """Concat critic, where we concat the inputs and use one MLP to output the value."""

    def __init__(self, dim, hidden_dim, layers, activation, **extra_kwargs):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp(dim * 2, hidden_dim, 1, layers, activation)

    def forward(self, x, y):
        batch_size = x.size(0)
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        scores = self._f(xy_pairs)
        return torch.reshape(scores, [batch_size, batch_size]).t()

class ConcatLipRKHS(nn.Module):
    def __init__(self, dim, hidden_dim, layers, activation,lip, **extra_kwargs):
        super(ConcatLipRKHS, self).__init__()
        # output is scalar score
        self.rkhs_layer = gaussian_perceptron(dim * 2, hidden_dim, 1, layers, activation, lip)
        self.num_feature = hidden_dim
        self.num_classes = 1
        self.p = nn.Parameter(self.get_lower_elements(self.num_classes, self.num_feature))

        self.mu = nn.Parameter(torch.rand((self.num_classes, self.num_feature)))

    def get_lower_elements(self, m, n):
        mat = torch.eye(n)

        indices = torch.triu_indices(n, n)
        out = mat[indices[0], indices[1]]
        out = (out.unsqueeze(0)).expand(m, -1)
        return out

    def forward(self, x, y):
        batch_size = x.size(0)
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        phi = self.rkhs_layer(xy_pairs)
        return phi, self.mu, self.p, self.num_feature

class ConcatLipFeatures(nn.Module):
    def __init__(self, dim, hidden_dim, layers, activation,lip, gamma =1, metric = 'rbf', D=500, mid_dim=5, g_lip =2, **extra_kwargs):
        super(ConcatLipFeatures, self).__init__()
        self.gamma = torch.FloatTensor([gamma])
        self.metric = metric
        self.D = D
        self.act = nn.ReLU()
        self.lin1 = spectral_norm( nn.Linear(hidden_dim, mid_dim), k = g_lip)
        self.lin2 = spectral_norm( nn.Linear(mid_dim, mid_dim), k = g_lip)
        self.lin3 = spectral_norm( nn.Linear(mid_dim, mid_dim), k = g_lip)
        self.lin4 = spectral_norm( nn.Linear(mid_dim, 1), k = g_lip)

        self.g = nn.Sequential(self.lin1,
                               self.act,
                               self.lin2,
                               self.act,
                               self.lin3,
                               self.act,
                               self.lin4
                               )
        # output of this layer is d dim features
        self.rkhs_layer = gaussian_perceptron(dim * 2, hidden_dim, 1, layers, activation, lip)


    def forward(self, x, y):
        batch_size = x.size(0)
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute features for each x_i, y_j pair.
        phi = self.rkhs_layer(xy_pairs)

        d = phi.shape[1]
        if self.metric == 'rbf':
            w = torch.sqrt(2 * self.gamma) * torch.randn(size=(self.D, d))
        w = w.to(x.device)
        psi = ((torch.matmul(phi, w.permute(1, 0)))) * (torch.sqrt(2 / torch.FloatTensor([self.D])).to(x.device))
        w_a = w  # torch.cat((w,u.permute(1,0)),1)
        g = self.g(w_a)
        f = (psi * g.permute(1, 0)).mean(1)
        g_norm = (g ** 2).mean()
        return f, g_norm




def log_prob_gaussian(x):
    return torch.sum(torch.distributions.Normal(0., 1.).log_prob(x), -1)
