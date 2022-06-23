import torch
from torch import nn


# use uncertainty value as weights to mixup feature
# we find that simply follow the traditional mixup setup
# to get mixup pairs can ensure good performance
def mixup_data(x, y, att, use_cuda=True):
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    att1 = att / (att + att[index])
    att2 = att[index] / (att + att[index])
    mixed_x = att1 * x + att2 * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, att1, att2


class RUL(nn.Module):
    def __init__(
        self,
        bbone,
        num_classes=7,
        drop_rate=0.4,
        inp_dim=512,
        out_dim=64,
        feature_size=7,
    ):
        super(RUL, self).__init__()

        self.drop_rate = drop_rate
        self.out_dim = out_dim
        self.features = bbone
        self.feature_size = feature_size

        self.mu = nn.Sequential(
            nn.BatchNorm2d(inp_dim, eps=2e-5, affine=False),
            nn.Dropout(p=self.drop_rate),
            nn.Flatten(),
            nn.Linear(inp_dim * feature_size * feature_size, self.out_dim),
            nn.BatchNorm1d(self.out_dim, eps=2e-5),
        )

        self.log_var = nn.Sequential(
            nn.BatchNorm2d(inp_dim, eps=2e-5, affine=False),
            nn.Dropout(p=self.drop_rate),
            nn.Flatten(),
            nn.Linear(inp_dim * feature_size * feature_size, self.out_dim),
            nn.BatchNorm1d(self.out_dim, eps=2e-5),
        )

    def train_forward(self, x, target):
        x = self.features(x)
        if self.feature_size == 1:
            x = torch.unsqueeze(torch.unsqueeze(x, 2), 3)
        mu = self.mu(x)
        logvar = self.log_var(x)

        mixed_x, y_a, y_b, att1, att2 = mixup_data(
            mu, target, logvar.exp().mean(dim=1, keepdim=True), use_cuda=True
        )
        return mixed_x, y_a, y_b, att1, att2

    def forward(self, x):
        x = self.features(x)
        output = self.mu(x)
        return output
