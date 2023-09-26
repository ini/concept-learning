import torch
from torch import Tensor



from torchvision.models import resnet18

model = resnet18(pretrained=True)
x = torch.randn(1, 3, 224, 224)
print(model(x).shape)

#32