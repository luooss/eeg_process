import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFuntion(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFuntion.apply(x, self.lambda_)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(62)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(310, 256)
        self.relu1 = nn.ReLU()
        # self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        a = self.flat(self.bn(x))
        a = self.relu1(self.fc1(a))
        a = self.fc2(a)
        return a


class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # if you want to apply additional operations in between layers, wirte them separately
        # or use nn.Sequential()
        self.fc1 = nn.Linear(128, 64)
        # self.dropout = nn.Dropout()
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 3)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        a = self.relu1(self.fc1(x))
        a = self.lsm(self.fc2(a))
        return a


class DANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.emotion_classifier = EmotionClassifier()
    
    def forward(self, x):
        a = self.feature_extractor(x)
        a = self.emotion_classifier(a)
        return a

class DomainClassifier(nn.Module):

    def __init__(self, lambda_):
        super().__init__()
        self.gradrev = GradientReversalLayer(lambda_=lambda_)
        self.fc1 = nn.Linear(128, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        a = self.gradrev(x)
        a = self.relu1(self.fc1(a))
        a = self.lsm(self.fc2(a))
        return a


class LSTM_Classification(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.bn = nn.BatchNorm1d(self.seq_len)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        self.lsm = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        out, _ = self.lstm(self.bn(x))
        out = out[:, -1, :]
        out = self.lsm(self.fc(out))
        return out
