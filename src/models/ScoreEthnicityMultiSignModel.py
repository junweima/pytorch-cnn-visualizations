import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import BaseModel

class LeakyClamp(nn.Module):
    def __init__(self, cap):
        super(LeakyClamp, self).__init__()
        self.cap = cap
        self.leakyrelu = nn.LeakyReLU(inplace=False)
        self.leakyrelu2 = nn.LeakyReLU(inplace=False)
    def forward(self, x):
        x = self.leakyrelu(x)
        x_ret = -self.leakyrelu2(-x + self.cap) + self.cap
        return x_ret

class ScoreEthnicityMultiSignModel(BaseModel):

    def __init__(self, n_ethn):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 150),
            nn.ReLU(inplace=True),
            nn.Linear(150, 1),
            LeakyClamp(7),
            # nn.Sigmoid()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.feature_size, 150),
            nn.ReLU(inplace=True),
            nn.Linear(150, 1),
            # nn.LeakyReLU(inplace=True),
            LeakyClamp(7),
            # nn.Sigmoid()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.feature_size, 150),
            nn.ReLU(inplace=True),
            nn.Linear(150, 1),
            # nn.LeakyReLU(inplace=True),
            LeakyClamp(8),
            # nn.Sigmoid()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(self.feature_size, 150),
            nn.ReLU(inplace=True),
            nn.Linear(150, 1),
            # nn.LeakyReLU(inplace=True),
            LeakyClamp(9),
            # nn.Sigmoid()
       ) 

        self.fc5 = nn.Sequential(
            nn.Linear(self.feature_size, 150),
            nn.ReLU(inplace=True),
            nn.Linear(150, 1),
            # nn.LeakyReLU(inplace=True),
            LeakyClamp(8),
            # nn.Sigmoid()
        )

        self.fc6 = nn.Sequential(
            nn.Linear(self.feature_size, 150),
            nn.ReLU(inplace=True),
            nn.Linear(150, 1),
            # nn.LeakyReLU(inplace=True),
            LeakyClamp(6),
            # nn.Sigmoid()
        )
        self.fc_ethn = nn.Linear(self.feature_size, n_ethn)

    def forward(self, x):
        x1 = self.features(x)
        x1 = x1.view(-1, self.feature_size)
        # x2 = self.features2(x)
        # x2 = x2.view(-1, self.feature_size)

        score1 = self.fc(x1)
        score2 = self.fc2(x1)
        score3 = self.fc3(x1)
        score4 = self.fc4(x1)
        score5 = self.fc5(x1)
        score6 = self.fc6(x1)
        # score = torch.clamp(score, min=0, max=9)

        score = torch.cat((score1, score2, score3, score4, score5, score6), 1)
        ethn = self.fc_ethn(x1)
        if not self.training:
            ethn = F.softmax(ethn, dim=1)
        return score, ethn

    def score_error(self, pred, target, score_scale):
        # put the mask for missing signs
        # pred[target == -1] = -1
        total = (target != -1).sum(dim=0).float()

        #abs_errs = (target - pred).abs() * score_scale
        abs_errs = (target - pred).abs() 
        abs_err_mean = abs_errs.sum(dim=0) / total
        percent_correct = ((abs_errs < 0.5) * (target != -1)).sum(dim=0).float() / total

        # if no sign in the batch
        abs_err_mean[total == 0] = 0
        percent_correct[total == 0] = 0

        abs_errs[target == -1] = -1

        return abs_err_mean, percent_correct, abs_errs

    def ethn_error(self, pred, target):
        _, top_ethn = pred.max(1)
        correct_ethn = (top_ethn == target).sum().item()
        accuracy_ethn = correct_ethn / len(target)
        return accuracy_ethn
