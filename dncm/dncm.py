import torch
import timm
from torch import nn
from kornia.geometry.transform import resize


class CustomLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rmse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.rmse(y_pred, y_true)) + self.mae(y_pred, y_true)


class RelativeRankingLoss(nn.Module):
    def __init__(self, lmbda=0.5):
        super(RelativeRankingLoss, self).__init__()
        self.L1 = nn.L1Loss()
        self.lmbda = lmbda

    def forward(self, predicted_scores, target_scores):
        # Sort predicted and target scores
        l1 = self.L1(predicted_scores, target_scores)

        sorted_pred, pred_indices = torch.sort(predicted_scores, descending=True)
        sorted_target = target_scores[pred_indices]

        # Get the highest, second highest, lowest, and second lowest scores
        q_max, q_max2, q_min, q_min2 = (
            sorted_pred[0],
            sorted_pred[1],
            sorted_pred[-1],
            sorted_pred[-2],
        )
        s_max, s_max2, s_min, s_min2 = (
            sorted_target[0],
            sorted_target[1],
            sorted_target[-1],
            sorted_target[-2],
        )

        self.margin1 = s_max2 - s_min
        self.margin2 = s_max - s_min2

        # Calculate absolute differences
        d = lambda x, y: torch.abs(x - y)  # noqa: E731

        # Compute triplet losses
        loss1 = torch.max(
            torch.tensor(0.0).to(predicted_scores.device),
            d(q_max, q_max2) - d(q_max, q_min) + self.margin1,
        )
        loss2 = torch.max(
            torch.tensor(0.0).to(predicted_scores.device),
            d(q_min2, q_min) - d(q_max, q_min) + self.margin2,
        )

        return l1 + self.lmbda * (loss1 + loss2)


class DNCM(nn.Module):
    # TODO : Add propoer linear projector
    def __init__(self, k) -> None:
        super().__init__()
        self.P = nn.Parameter(torch.empty((3, k)), requires_grad=True)
        self.Q = nn.Parameter(torch.empty((k, 3)), requires_grad=True)

        self.fc_score = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.ReLU(),
        )
        self.fc_weight = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid(),
        )
        torch.nn.init.kaiming_normal_(self.P)
        torch.nn.init.kaiming_normal_(self.Q)
        self.k = k

    def forward(self, I, T):
        bs, _, H, W = I.shape
        x = torch.flatten(I, start_dim=2).transpose(1, 2)
        out = x @ self.P @ T.view(bs, self.k, self.k) @ self.Q
        out = out.transpose(1, 2).view(bs, 3, H, W)
        score = torch.tensor([]).cuda()
        for i in range(out.shape[0]):
            f = self.fc_score(out[i])
            w = self.fc_weight(out[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score


class Encoder(nn.Module):
    def __init__(self, sz, k) -> None:
        super().__init__()
        # self.backbone = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA", regressor_dataset="kadid10k")
        self.backbone = timm.create_model(
            "vit_large_patch16_384.augreg_in21k_ft_in1k", pretrained=True, num_classes=0
        )
        self.backbone.eval()
        self.D = nn.Linear(in_features=1024, out_features=k * k)
        self.sz = sz

    def forward(self, I):
        I_theta = resize(I, self.sz, interpolation="bilinear")
        # I_theta_half = resize(I_theta, (I_theta.shape[0], I_theta.shape[1]), interpolation='bilinear')
        with torch.no_grad():
            out = self.backbone(I_theta)
        d = self.D(out)
        return d


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


class MobileViTv2Attention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
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
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
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
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        """
        if reference is None:
            reference = input
        i = self.fc_i(input)  # (bs,nq,1)
        weight_i = torch.softmax(i, dim=1)  # bs,nq,1
        context_score = weight_i * self.fc_k(input)  # bs,nq,d_model
        context_vector = torch.sum(context_score, dim=1, keepdim=True)  # bs,1,d_model
        v = self.fc_v(reference) * context_vector  # bs,nq,d_model
        out = self.fc_o(v)  # bs,nq,d_model

        return out


if __name__ == "__main__":
    k = 32
    I = torch.rand((1, 3, 512, 512)).cuda()
    net = DNCM(k).cuda()
    E = Encoder((384, 384), k).cuda()
    d = E(I)
    # print(d.shape)
    out = net(I, d)
    print(out)
