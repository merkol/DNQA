import torch
import timm
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
        # out = torch.sigmoid(out)
        return out
    
class DeNIM_StyleSwap_to_Canon(nn.Module):
    def __init__(self, k, ch: int = 3) -> None:
        super().__init__()
        self.ch = ch
        self.inp_proj = nn.Parameter(torch.empty((ch, k * k)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.inp_proj)
        self.res_proj = nn.Parameter(torch.empty((ch, 3)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.res_proj)
        self.vit_v2_attn = MobileViTv2Attention(k * k)
        self.out_proj = nn.Parameter(torch.empty((k * k, 3)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.out_proj)
        self.silu = torch.nn.SiLU()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, I, T):
        bs, _, H, W = I.shape
        I_flat = torch.flatten(I, start_dim=2).transpose(1, 2)
        feat_proj = I_flat @ self.inp_proj
        residual_proj = I_flat @ self.res_proj
        out = self.vit_v2_attn(feat_proj, T.unsqueeze(1).repeat(1, H * W, 1))
        out = out @ self.out_proj
        out = out + residual_proj
        out = self.silu(out)
        out = self.global_avg_pool(out)
        out = out.flatten(start_dim=1)
        # out = torch.sigmoid(out)
        return out

class Encoder(nn.Module):
    def __init__(self, sz, k, backbone_arch) -> None:
        super().__init__()
        self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.backbone = timm.create_model(backbone_arch, pretrained=True)
        self.D = nn.Linear(in_features=1000, out_features=k*k)
        self.sz = sz
        
    def forward(self, I):
        I_theta = resize(I, self.sz, interpolation='bilinear')
        # with torch.no_grad():
        out = self.backbone(self.normalizer(I_theta))
        d = self.D(out)
        return d
    
    
class MobileViTv2Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model, 1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.d_model = d_model
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, input, reference=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        '''
        if reference is None:
            reference = input
        i = self.fc_i(input) #(bs,nq,1)
        weight_i = torch.softmax(i, dim=1) #bs,nq,1
        context_score = weight_i * self.fc_k(input) #bs,nq,d_model
        context_vector = torch.sum(context_score, dim=1, keepdim=True) #bs,1,d_model
        v = self.fc_v(reference) * context_vector #bs,nq,d_model
        out = self.fc_o(v) #bs,nq,d_model

        return out
        
    
if __name__ == "__main__":
    k = 32 
    I = torch.rand((8, 3, 968, 3840)).cuda()
    net = DNCM(k).cuda()
    E = Encoder((252, 252), k).cuda()
    d = E(I)
    out = net(I, d)
    print(out.shape)