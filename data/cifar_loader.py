from torchvision import datasets, transforms

def get_cifar_dataloader(dataset="cifar10"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset.lower() == "cifar10":
        train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    return train_data, test_data
