# Entry point for training
from server.server import FederatedServer

if __name__ == "__main__":
    server = FederatedServer(dataset="cifar10", num_clients=5, num_rounds=25)
    server.train()