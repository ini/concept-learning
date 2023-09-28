import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor



def _pad_along_last_axis(X, m):
    """Pad the data for computing the rolling window difference."""
    # scales a  bit better than method in _vasicek_like_entropy
    shape = list(X.shape)
    shape[-1] = m
    Xl = torch.broadcast_to(X[..., [0]], shape)  # [0] vs 0 to maintain shape
    Xr = torch.broadcast_to(X[..., [-1]], shape)
    return torch.cat((Xl, X, Xr), axis=-1)

def _vasicek_entropy(X, m):
    """Compute the Vasicek estimator as described in [6] Eq. 1.3."""
    n = X.shape[-1]
    X = _pad_along_last_axis(X, m)
    differences = X[..., 2 * m:] - X[..., : -2 * m:]
    logs = torch.log(n/(2*m) * torch.maximum(differences, torch.tensor(1e-6)))
    return torch.mean(logs, axis=-1)


from torch.distributions import MultivariateNormal
m = MultivariateNormal(torch.zeros(28*28), torch.eye(28*28))

# def entropy(r, c, eps=1e-6):
#     x = torch.cat([r, c], dim=-1)
#     mu = x.mean(dim=0)
#     var = x.var(dim=0)
#     var = torch.maximum(var, torch.tensor(eps))
#     return 1e-3 * (var.log() + (x - mu)**2 / var).sum(dim=-1).mean()

def kl_from_unit_gaussian(x):
    dist = MultivariateNormal(torch.zeros(x.shape[-1]), torch.eye(x.shape[-1]))
    #x = x.clone()
    x = (x - x.mean(dim=0)) / torch.maximum(x.std(dim=0), torch.tensor(1e-6))
    return (28*28 / dist.log_prob(x)).mean()

    p = torch.ones(len(x)) / len(x)
    log_q = m.log_prob(x)
    return (p * p.log() / log_q).sum()

    mu = x.mean(dim=0)
    var = x.var(dim=0)
    return -0.5 * (var.log() - var - mu**2 + 1).sum(dim=-1).mean()

import math
def differential_entropy(X):
    #return -kl_from_unit_gaussian(X)
    X = torch.moveaxis(X, 0, -1)
    X, _ = torch.sort(X, dim=-1)
    window_length = math.floor(math.sqrt(X.shape[-1]) + 0.5)

    H = _vasicek_entropy(X, window_length).mean()
    #import tqdm
    #tqdm.tqdm.write(str(H.item()))
    return H #.exp()
    return _vasicek_entropy(X, window_length).sum()



class EntropyLayer(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch_dim, input_dim)
            Input samples
        """
        self.inputs = x
        return x

    def entropy(self) -> Tensor:
        """
        Return the entropy of the last input batch.
        """
        x = self.inputs.view(-1, self.inputs.shape[-1]).clone()
        # x -= x.mean(dim=0)
        # x /= torch.maximum(x.std(dim=0), torch.tensor(1e-6))
        return differential_entropy(x)
        mu = x.mean(dim=0).unsqueeze(0)
        var = x.var(dim=0).unsqueeze(0)
        return (var.log() + (x - mu)**2 / var).sum(dim=-1).mean()


# ccc = EntropyLayer()

x = torch.randn(64, 28*28)
print(differential_entropy(x), -kl_from_unit_gaussian(x))
X = torch.moveaxis(x.clone(), 0, -1)
X, _ = torch.sort(X, dim=-1)
window_length = math.floor(math.sqrt(X.shape[-1]) + 0.5)
print(_vasicek_entropy(X, window_length).shape)

# x = torch.rand(64, 28*28)
# print(differential_entropy(x), -kl_from_unit_gaussian(x))

# x = torch.ones(100, 28*28)
# print(differential_entropy(x), -kl_from_unit_gaussian(x))

# x[32:] = 1
# print(differential_entropy(x), -kl_from_unit_gaussian(x))


# x = torch.rand(10000, 8)
# ccc(x)
# print(ccc.entropy())

# x = torch.randn(10000, 8) ** 2
# ccc(x)
# print(ccc.entropy())

# x = torch.rand(10000, 8) ** 2
# ccc(x)
# print(ccc.entropy())

# x = torch.tensor([1, 1, 1, 2, 4, 4, 4, 4]).float().unsqueeze(-1)
# ccc(x)
# print(ccc.entropy())

# x = torch.zeros(1000, 8)
# x[:500] = 1
# x[:1000] = 20000
# ccc(x)
# print(ccc.entropy())


# def thing():
#     n = 0
#     def f():
#         nonlocal n
#         n += 1
#         print(n)
    
#     for i in range(10):
#         f()

# thing()