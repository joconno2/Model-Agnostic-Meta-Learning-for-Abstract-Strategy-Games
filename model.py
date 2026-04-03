import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModule(nn.Module):
    '''
    Convulutions 3x3, Batch Normalization, ReLu (same module as the paper)
    '''

    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)
    
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
            for i, m in enumerate(self.mods, start = 1):
                x = m(x)
                if debug:
                    print(f" after module{i}: tuple(x.shape)")
                return x #[6, 64, 9, 9]
            
class ChessPolicyValueNet(nn.Module):
    "Shared trunk for both the value and policy head"
    def __init__(self, in_channels: int, n_actions: int = 20480, trunk_hidden: int = 64):
        super().__init__()
        self.trunk = Trunk(in_channels, hidden=trunk_hidden, n_modules=4)
        feat_dim = trunk_hidden * 9 * 9
        #Policy head (linear)
        self.policy_fc = nn.Linear(feat_dim, n_actions)

        #Value head
        self.value_fc1 = nn.Linear(feat_dim, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x, debug: bool = False):
        h = self.trunk(x, debug = debug)
        h = h.flatten(1)

        if debug:
            print(f" flattened: {tuple(h.shape)}")
            
        logits = self.policy_fc(h)
        v = F.relu(self.value_fc1(h), inplace = True)
        v = self.value_fc2(v).squeeze(-1)

        if debug:
            print(f" policy logits: {tuple(logits.shape)}")
            print(f" value pref: {tuple(v.shape)}")

        return logits, v

def print_architecture_graph(in_channels: int, n_actions: int = 20480):
    """
    Prints an easy-to-read architecture graph with shapes.
    """
    print("\n=== Architecture Graph (Paper-style trunk, no pooling) ===")
    print(f"Input: X [B, {in_channels}, 9, 9]")
    print("  │")
    print("  ├─ Trunk: 4×(Conv3×3 -> BN -> ReLU), 64 filters")
    print("  │     module1: [B, 64, 9, 9]")
    print("  │     module2: [B, 64, 9, 9]")
    print("  │     module3: [B, 64, 9, 9]")
    print("  │     module4: [B, 64, 9, 9]")
    print("  │")
    print("  ├─ Flatten: [B, 64*9*9] = [B, 5184]")
    print("  │")
    print(f"  ├─ Policy head: Linear(5184 → {n_actions}) => logits [B, {n_actions}]")
    print("  │       Loss: CrossEntropy(logits, action_id)")
    print("  │")
    print("  └─ Value head: Linear(5184 → 128) → ReLU → Linear(128 → 1) => v [B]")
    print("          Loss: MSE(v, z)")
    print("=========================================================\n")
