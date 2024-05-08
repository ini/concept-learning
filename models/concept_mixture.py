from torch import nn
import torch


class ConceptMixture(nn.Module):
    def __init__(self, concept_dim, residual_dim, target_network, mixer):
        super(ConceptMixture, self).__init__()
        self.concept_dim = concept_dim
        self.residual_dim = residual_dim
        self.target_network = target_network
        self.mixer = mixer

    def forward(self, x, concepts=None, which=None):
        if type(x) == tuple:
            x, concepts, which = x

        if which is None:
            out = self.target_network(x)
            return out

        new_concepts = x[..., : self.concept_dim]
        residual = x[..., self.concept_dim :]
        target_network_x = torch.cat([new_concepts * (~which), residual], dim=-1)
        mixer_x = new_concepts * which

        if type(self.target_network) == nn.Linear:
            out = self.target_network(target_network_x)
            out += self.mixer(mixer_x)
        else:
            first_layer_out = self.target_network[:1](target_network_x)
            first_layer_out += self.mixer(mixer_x)
            out = self.target_network[1:](first_layer_out)
        return out
