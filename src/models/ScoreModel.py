import torch.nn as nn
from models.BaseModel import BaseModel


class ScoreModel(BaseModel):

    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.feature_size)
        score = self.fc(x)
        return score

    def score_error(self, pred, target, score_scale):
        absolute_error = (target - pred).abs() * score_scale
        percent_correct = len(absolute_error[absolute_error < 0.5]) / len(absolute_error)
        return absolute_error.mean(), percent_correct
