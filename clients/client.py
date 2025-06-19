import torch
import torch.nn as nn
import torch.nn.functional as F
from models.autoencoder import SimpleAutoencoder
import copy

class Client:
    def __init__(self, id, model, data, device):
        self.id = id
        self.model = model.to(device)
        self.device = device
        self.data = data

    def update_model(self, global_weights):
        self.model.load_state_dict(copy.deepcopy(global_weights))

    def local_train(self, epochs=1, lr=0.01):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        loader = torch.utils.data.DataLoader(self.data, batch_size=64, shuffle=True)
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()
        return self.model.state_dict()

    def train_autoencoder(self, epochs=3, lr=0.001):
        self.autoencoder = SimpleAutoencoder().to(self.device)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        loader = torch.utils.data.DataLoader(self.data, batch_size=64, shuffle=True)
        self.autoencoder.train()
        for epoch in range(epochs):
            for x, _ in loader:
                x = x.to(self.device)
                output = self.autoencoder(x)
                loss = loss_fn(output, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def detect_anomalies(self, threshold=0.02):
        self.autoencoder.eval()
        anomalies = []
        loader = torch.utils.data.DataLoader(self.data, batch_size=64, shuffle=False)
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                output = self.autoencoder(x)
                loss = ((x - output) ** 2).view(x.size(0), -1).mean(dim=1)
                batch_anomalies = (loss > threshold).cpu().numpy()
                anomalies.extend(batch_anomalies)
        return anomalies
