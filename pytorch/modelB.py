import pretrainedmodels
import torch.nn as nn


class modelA(nn.Module):

    def __init__(self, num_classes, inference):
        super(modelA, self).__init__()

        self.encoder = pretrainedmodels.__dict__["squeezenet1_1"](num_classes=1000, pretrained='imagenet')
        self.global_pool = nn.AvgPool2d(13, stride=1)
        self.classifier = nn.Sequential()
        self.classifier.add_module('proj', nn.Linear(512, num_classes))
        if inference:
            self.classifier.add_module('softmax', nn.Softmax())

        # self.freeze_module([self.encoder])

    def forward(self, batch):
        rgb = batch['rgb']

        features = self.encoder.features(rgb)
        pooled = self.global_pool(features).view(features.size(0), -1)
        probs = self.classifier(pooled)
        return probs

    def freeze_module(self, module_list: list):
        for mod in module_list:
            for param in mod.parameters():
                param.requires_grad = False
