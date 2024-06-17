import torch
from torch import nn
from kornia.geometry.transform import resize
from kornia.enhance.normalize import Normalize


class DNCM(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()
        self.P = nn.Parameter(torch.empty((3, k)), requires_grad=True)
        self.Q = nn.Parameter(torch.empty((k, 3)), requires_grad=True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        torch.nn.init.kaiming_normal_(self.P)
        torch.nn.init.kaiming_normal_(self.Q)
        self.k = k
        
    def forward(self, I, T):
        bs, _, H, W = I.shape
        x = torch.flatten(I, start_dim=2).transpose(1, 2)
        out = x @ self.P @ T.view(bs, self.k, self.k) @ self.Q
        out = self.global_avg_pool(out)
        out = out.flatten(start_dim=1)
        out = torch.sigmoid(out)
        return out
    

class Encoder(nn.Module):
    def __init__(self, sz, k) -> None:
        super().__init__()
        self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.D = nn.Linear(in_features=768, out_features=k*k)
        self.sz = sz
        
    def forward(self, I):
        I_theta = resize(I, self.sz, interpolation='bilinear')
        with torch.no_grad():
            out = self.backbone(self.normalizer(I_theta))
        d = self.D(out)
        return d
        
    
if __name__ == "__main__":
    k = 32 
    I = torch.rand((8, 3, 968, 3840)).cuda()
    net = DNCM(k).cuda()
    E = Encoder((252, 252), k).cuda()
    d = E(I)
    out = net(I, d)
    print(out.shape)