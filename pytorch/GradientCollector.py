import torch
from torch import nn

SQUARED_SUFFIX = '_squared'
RELATIVE_SQUARED_SUFFIX = '_rel'


class GradientCollector:
    """
    Collects (averages) gradients during training
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.collector = {}
        self.num_collected = 0

    def collect(self, model: nn.Module) -> None:
        """
        Call one time each forward pass
        :param model: model to get gradients from
        """
        eps = 1e-8

        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                g = param.grad.data
                if name not in self.collector:
                    self.collector[name] = torch.zeros(g.shape).to(self.device)
                    self.collector[name + SQUARED_SUFFIX] = torch.zeros(g.shape).to(self.device)
                    self.collector[name + RELATIVE_SQUARED_SUFFIX] = torch.zeros(g.shape).to(self.device)
                self.collector[name] += g
                squared = g ** 2
                self.collector[name + SQUARED_SUFFIX] += squared

                # calculating average gradient
                relative = squared / (param.data ** 2 + eps)
                self.collector[name + RELATIVE_SQUARED_SUFFIX] += relative

        self.num_collected += 1

    def get_and_reset(self) -> dict:
        """
        Returns averaged gradients collected so far
        :return: averaged numpy array: dictionary<layer_name, torch.Tensor>
        """

        collected = self.collector
        num = torch.Tensor([self.num_collected]).to(self.device)
        del self.collector
        self.collector = {}
        self.num_collected = 0

        # normalizing
        normalized = {}
        for key, val in collected.items():
            normalized[key] = (val / num).cpu().numpy()

        return normalized


