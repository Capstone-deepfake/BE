# backend/deepfake/modelNet.py

import torch.nn as nn

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # --- 예시 구조(실제 구조와 완전히 일치시켜야 합니다) ---
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        # (224→112) 가정. 실제 크기를 맞춰야 합니다.
        self.fc = nn.Linear(16 * 112 * 112, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
