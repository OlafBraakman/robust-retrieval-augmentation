import torch
import torch.nn as nn

class RandomizedSmoothing(nn.Module):

    def __init__(self, config, std_transform=torch.tensor([0.2686, 0.2613, 0.2758])):
        super().__init__()

        self.noise_std = config['noise_std']
        self.std_transform = std_transform

    def forward(self, x):
        # Unlike the original implementation we are not concerned about certification in the pixel domain.
        # So apply noise in the transformed domain, but make sure to take std transformation into account
        return x + (torch.randn_like(x) * self.noise_std) / self.std_transform.view(1, 3, 1, 1).to(x.device)
