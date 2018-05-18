import torch.nn as nn
from torchvision import models


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        resnet.avgpool = nn.AdaptiveMaxPool2d(1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        #self.features2 = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_size = 2048

    def forward(self, x):
        raise NotImplementedError('Must subclass BaseModel and implement the forward pass')

    def finetune_parameters(self):
        return self.features.parameters()

    def new_parameters(self):
        ignored_params = list(map(id, self.finetune_parameters()))
        new_params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        return new_params
