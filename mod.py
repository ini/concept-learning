import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor




class SegmentedLinear2(nn.Linear):

    def __init__(self, in_features: int, out_features: int, N: int = 1, **kwargs):
        super().__init__(in_features * N, out_features * N, **kwargs)
        self.N = N
        self.weight_factor = torch.zeros_like(self.weight.data)
        for i in range(N):
            self.weight_factor[
                i * out_features : (i + 1) * out_features,
                i * in_features : (i + 1) * in_features,
            ] = 1
        self.weight_factor[:out_features, :in_features] = 1

    def forward(self, input: Tensor) -> Tensor:
        out = F.linear(
            input.view(*input.shape[:-2], -1),
            self.weight * self.weight_factor,
            self.bias,
        )
        return out.view(*input.shape[:-1], -1)


class SegmentedLinear(nn.Module):

    def __init__(self, input_dim, output_dim, N=1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(N)])

    def forward(self, x):
        x = x.view(-1, len(self.layers), x.shape[-1])
        return torch.stack([self.layers[i](x[:, i]) for i in range(len(self.layers))], dim=1)



class CCC(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Parameters
        ----------
        dim : int
            Dimension of input samples
        hidden_dim : int
            Dimension of hidden layers in variational approximation network
        """
        super().__init__()
        self.input_dim = input_dim

        self.p_mu = nn.Sequential(
            SegmentedLinear2(1, hidden_dim, N=input_dim), nn.ReLU(),
            SegmentedLinear2(hidden_dim, hidden_dim, N=input_dim), nn.ReLU(),
            SegmentedLinear2(hidden_dim, input_dim, N=input_dim),
        )
        self.p_log_var = nn.Sequential(
            SegmentedLinear2(1, hidden_dim, N=input_dim), nn.ReLU(),
            SegmentedLinear2(hidden_dim, hidden_dim, N=input_dim), nn.ReLU(),
            SegmentedLinear2(hidden_dim, input_dim, N=input_dim), nn.Tanh(),
        )
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def log_likelihood(self, x: Tensor) -> Tensor:
        """
        Return matrix of (unnormalized) pairwise conditional log-likelihoods.

        Parameters
        ----------
        x : Tensor of shape (batch_dim, input_dim)
            Input samples

        Returns
        -------
        Q : Tensor of shape (batch_dim, input_dim, input_dim)
            Matrix of conditional log-likelihoods, where Q[n, i, j]
            is the log-likelihood of x[n, i] given x[n, j].
        """
        x = x.unsqueeze(-1)
        mu, log_var = self.p_mu(x), self.p_log_var(x)

        log_var = log_var * 0 + 1
        #print("LV", log_var.mean().item())

        #losses = (mu - x) ** 2
        #print(losses.mean(dim=0))
        # print("losses", losses.mean().item(), losses.std().item(), losses.min().item(), losses.max().item())
        return -0.5 * ((mu - x)**2 / log_var.exp() + log_var)

    def mutual_information(self, x: Tensor) -> Tensor:
        """
        Return an estimated upper bound for the mutual information matrix.

        Parameters
        ----------
        x : Tensor of shape (batch_dim, input_dim)
            Input samples

        Returns
        -------
        M : Tensor of shape (input_dim, input_dim)
            Mutual information upper bound matrix, where M[i, j] is an estimated
            upper bound on the mutual information between x[:, i] and x[:, j].
        """
        Q = self.log_likelihood(x)

        # Positive samples
        positive = Q

        # Negative samples
        negative = Q.mean
        negative = Q.view(Q.shape[0], -1)
        negative = negative[
            torch.argsort(torch.rand(*negative.shape), dim=0),
            torch.arange(negative.shape[1]),
        ].view(Q.shape)

        # Contrastive log-ratio upper bound
        return (positive - negative).mean(dim=0)

    def variational_loss(self, x: Tensor) -> Tensor:
        """
        Loss for variational network used to approximate
        conditional log-likelihoods.

        Parameters
        ----------
        x : Tensor of shape (batch_dim, input_dim)
            Input samples
        """
        return -self.log_likelihood(x).mean()

    def disentanglement_loss(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch_dim, input_dim)
            Input samples
        """
        return self.mutual_information(x).mean()

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch_dim, input_dim)
            Input samples
        """
        return x






def get_base_model(input_dim=28*28, output_dim=10, hidden_dim=128):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class MyConceptModel(nn.Module):

    def __init__(self, concept_dim: int):
        super().__init__()
        self.concept_dim = concept_dim
        self.concept_network = get_base_model(output_dim=concept_dim)
        self.predictor_network = get_base_model(input_dim=concept_dim)

    def forward(self, x):
        concept_preds = self.concept_network(x)
        out = self.predictor_network(concept_preds)
        return concept_preds, out

    def concept_predictions(self, x):
        return self.concept_network(x)
        







from data_real import pitfalls_decorrelation_dataset
from utils import train_multiclass_classification, concepts_preprocess_fn


train_loader, test_loader, concept_dim = pitfalls_decorrelation_dataset()

model = MyConceptModel(concept_dim)
my_mod = CCC(input_dim=concept_dim)
my_mod_optimizer = torch.optim.Adam(my_mod.parameters(), lr=0.001)

def variational_training_step(model, epoch, batch_index, batch):
    (X, concepts), y = batch
    my_mod_optimizer.zero_grad()
    with torch.no_grad():
        concept_preds = model.concept_network(X)

    variational_loss = my_mod.variational_loss(concept_preds)
    variational_loss.backward()
    my_mod_optimizer.step()

def loss_fn(data, output, target):
    X, concepts = data
    concept_preds, output = output
    concept_loss = F.binary_cross_entropy(concept_preds, concepts)
    disentanglement_loss = my_mod.disentanglement_loss(concept_preds)
    prediction_loss = F.cross_entropy(output, target)
    return concept_loss + disentanglement_loss + prediction_loss

model = train_multiclass_classification(
    model,
    train_loader,
    test_loader=test_loader,
    num_epochs=20,
    preprocess_fn=concepts_preprocess_fn,
    callback_fn=variational_training_step,
)

correct = 0
total = 0
with torch.no_grad():
    for (X, concepts), y in test_loader:
        activations = model.cw_layer(
            model.concept_network(X)[..., None, None])[..., :model.concept_dim, 0, 0]

        predicted_concept_idx = torch.topk(activations, 3).indices.sort().values
        results = torch.stack([
            concepts[i, predicted_concept_idx[i]] for i in range(len(concepts))])
        correct += results.sum()
        total += results.nelement()

accuracy = 100 * correct / total
print(f"Final Concept Accuracy on the test dataset: {accuracy:.2f}%")






### Calculate Intra-Concept Leakage

for concept_index in range(model.concept_dim):
    other_idx = [i for i in range(model.concept_dim) if i != concept_index]
    model_c = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, len(other_idx)))
    
    def transform(batch):
        (X, concepts), y = batch
        with torch.no_grad():
            inputs = model.concept_predictions(X)[..., [concept_index]]
            targets = concepts[..., other_idx].argmax(dim=-1)
            return inputs, targets
        
    model_c = train_multiclass_classification(
        model_c, train_loader, test_loader=test_loader, preprocess_fn=transform, num_epochs=5)
