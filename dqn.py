import torch
import numpy as np
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DQN(nn.Module):
    def __init__(self, batch_size) -> None:
        super().__init__()

        self.in_features = 16 # 4x4=16
        self.out_actions = 4 # 4 possible actions in action space: UDLR

        self.net = nn.Sequential(
            # nn.Linear(32, 64),
            # nn.BatchNorm1d(4),
            # nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.BatchNorm1d(4),
            # nn.ReLU(),
            # nn.Linear(64, self.out_actions)

            nn.Conv2d(1, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Flatten(),
            nn.Linear(512, 32),
            nn.ReLU(),
    
        )
    
    def forward(self, x):
        # x = x.permute(1, 2, 0) # 4x4 @ 32
        # x = x.flatten()
        # print(x.shape) # 1x1x4x4
        result = self.net(x)
        # print(result)
        return result

    def act(self, state):
        state = state.permute(1, 0, 2) # 4x1x4 -> 1x4x4
        state = state.unsqueeze(0) # 3D -> 4D by adding dimension to the 0th axis, yielding 1x1x4x4
        q_values = self(state) # Q-values for every possible action in this state

        best_action_idx = torch.argmax(q_values, dim=1)[0]
        best_action = best_action_idx.detach().item()
        # print(best_action)
        return best_action
        