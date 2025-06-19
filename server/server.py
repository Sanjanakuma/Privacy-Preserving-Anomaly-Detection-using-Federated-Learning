import torch
import copy
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
from collections import defaultdict
import numpy as np
from models.cnn import SimpleCNN
from clients.client import Client
from data.cifar_loader import get_cifar_dataloader

class FederatedServer:
    def __init__(self, dataset="cifar10", num_clients=5, num_rounds=25):
        # Clear old files for fresh run
        if os.path.exists("weight_trends.png"):
            os.remove("weight_trends.png")
        if os.path.exists("client_weight_log.xlsx"):
            os.remove("client_weight_log.xlsx")

        self.log_rows = []  # store row dictionaries
        self.accuracy_log = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = SimpleCNN(num_classes=10 if dataset == "cifar10" else 100).to(self.device)

        train_data, test_data = get_cifar_dataloader(dataset)
        print(f"Train set size: {len(train_data)}")
        print(f"Test set size: {len(test_data)}")

        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
        self.clients = self._init_clients(train_data, num_clients)
        self.num_rounds = num_rounds
        self.weight_stats = {client.id: [] for client in self.clients}

    def _init_clients(self, train_data, num_clients):
        seed = int(time.time())
        random.seed(seed)
        torch.manual_seed(seed)
        split_sizes = [len(train_data) // num_clients] * num_clients
        data_split = torch.utils.data.random_split(train_data, split_sizes)
        return [Client(i, copy.deepcopy(self.global_model), data_split[i], self.device) for i in range(num_clients)]

    
    '''def _init_clients(self, train_data, num_clients,shards_per_client=2):
        # Extract labels from CIFAR dataset
        targets = torch.tensor(train_data.targets)
        data_by_class = defaultdict(list)

        for idx, label in enumerate(targets):
            data_by_class[label.item()].append(idx)

        # Sort and split indices into shards
        shards = []
        for label, indices in data_by_class.items():
            np.random.shuffle(indices)
            shard_size = len(indices) // shards_per_client
            for i in range(shards_per_client):
                shard = indices[i * shard_size: (i + 1) * shard_size]
                shards.append(shard)

        np.random.shuffle(shards)
        assert len(shards) >= num_clients * shards_per_client

        # Assign shards to clients
        client_data = []
        for i in range(num_clients):
            shard_indices = []
            for _ in range(shards_per_client):
                shard_indices.extend(shards.pop())
            subset = torch.utils.data.Subset(train_data, shard_indices)
            client_data.append(subset)

        return [Client(i, copy.deepcopy(self.global_model), client_data[i], self.device) for i in range(num_clients)]'''


    def aggregate(self, client_weights):
        avg_weights = copy.deepcopy(client_weights[0])
        for key in avg_weights.keys():
            for i in range(1, len(client_weights)):
                avg_weights[key] += client_weights[i][key]
            avg_weights[key] /= len(client_weights)
        self.global_model.load_state_dict(avg_weights)

    def evaluate_global_model(self):
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.global_model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total

    def plot_weight_stats(self):
        for client_id, stats in self.weight_stats.items():
            plt.plot(stats, label=f"Client {client_id}")
        plt.xlabel("Round")
        plt.ylabel("Mean of conv1.weight")
        plt.title("Mean of conv1.weight Across Rounds")
        plt.legend()
        plt.grid(True)
        plt.savefig("weight_trends.png")
        plt.show()

    def plot_accuracy(self):
        plt.plot(self.accuracy_log, marker='o')
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.title("Global Accuracy Across Rounds")
        plt.grid(True)
        plt.savefig("accuracy_plot.png")
        plt.show()

    def train(self):
        for r in range(self.num_rounds):
            print(f"\n--- Round {r+1} ---")
            weights = []

            for client in self.clients:
                client_weights = client.local_train()
                weights.append(client_weights)

                # Log stats for Excel
                stats = {
                    "Round": r+1,
                    "Client": client.id,
                }
                for name, tensor in client_weights.items():
                    stats[f"{name}_mean"] = tensor.mean().item()
                    stats[f"{name}_std"] = tensor.std().item()
                    if name == "features.0.weight":
                        self.weight_stats[client.id].append(tensor.mean().item())
                self.log_rows.append(stats)

            self.aggregate(weights)

            for client in self.clients:
                client.update_model(self.global_model.state_dict())

            print("Detecting anomalies...")
            all_anomalies = []
            for client in self.clients:
                client.train_autoencoder()
                anomalies = client.detect_anomalies(threshold=0.5)
                all_anomalies.extend(anomalies)

            anomaly_rate = sum(all_anomalies) / len(all_anomalies)
            print(f"Estimated Anomaly Rate (Round {r+1}): {anomaly_rate:.2%}")

            acc = self.evaluate_global_model()
            self.accuracy_log.append(acc)
            print(f"Global Model Accuracy (Round {r+1}): {acc:.2%}")

        self.plot_weight_stats()
        self.plot_accuracy()
        pd.DataFrame(self.log_rows).to_excel("client_weight_log.xlsx", index=False)
        print("Training complete. Logs saved to 'client_weight_log.xlsx'")

