import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import BaseModel


class ScoreEthnicityModel(BaseModel):

    def __init__(self, n_ethn):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

        self.fc_ethn = nn.Linear(self.feature_size, n_ethn)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.feature_size)
        score = self.fc(x)
        ethn = self.fc_ethn(x)
        if not self.training:
            ethn = F.softmax(ethn, dim=1)
        return score, ethn

    def score_error(self, pred, target, score_scale):
        absolute_error = (target - pred).abs() * score_scale
        percent_correct = len(absolute_error[absolute_error < 0.5]) / len(absolute_error)
        return absolute_error.mean(), percent_correct

    def ethn_error(self, pred, target):
        _, top_ethn = pred.max(1)
        correct_ethn = (top_ethn == target).sum().item()
        accuracy_ethn = correct_ethn / len(target)
        return accuracy_ethn
