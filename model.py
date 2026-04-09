import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    
    #3x3 Convolution + ReLU (NO BatchNorm)

    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


class Trunk(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, n_modules: int = 4):
        super().__init__()
        mods = []
        ch = in_channels
        for _ in range(n_modules):
            mods.append(ConvModule(ch, hidden))
            ch = hidden
        self.mods = nn.ModuleList(mods)

    def forward(self, x, debug: bool = False):
        if debug:
            print(f" trunk input: {tuple(x.shape)}")

        for i, m in enumerate(self.mods, start=1):
            x = m(x)
            if debug:
                print(f" after module{i}: {tuple(x.shape)}")

        return x


class ChessPolicyValueNet(nn.Module):
    
    #Shared trunk for both policy and value heads.
    def __init__(self, in_channels: int, n_actions: int = 20480, trunk_hidden: int = 64):
        super().__init__()

        self.trunk = Trunk(in_channels, hidden=trunk_hidden, n_modules=4)
        feat_dim = trunk_hidden * 9 * 9

        # Policy head
        self.policy_fc = nn.Linear(feat_dim, n_actions)

        # Value head
        self.value_fc1 = nn.Linear(feat_dim, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x, debug: bool = False):
        h = self.trunk(x, debug=debug)
        h = h.flatten(1)

        if debug:
            print(f" flattened: {tuple(h.shape)}")

        logits = self.policy_fc(h)

        v = F.relu(self.value_fc1(h), inplace=True)
        v = self.value_fc2(v).squeeze(-1)

        if debug:
            print(f" policy logits: {tuple(logits.shape)}")
            print(f" value pred: {tuple(v.shape)}")

        return logits, v