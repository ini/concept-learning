import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset



# Define a function to filter dataset by class
def filter_dataset_by_class(dataset, target_classes):
    indices = [i for i in range(len(dataset)) if dataset.targets[i] in list(target_classes)]
    return indices

# Define a function to create a new dataset with images and parity targets
def create_parity_dataset(original_dataset, target_classes):
    target_indices = filter_dataset_by_class(original_dataset, target_classes)
    dataset = []

    for idx in target_indices:
        image, target = original_dataset[idx]
        concepts = torch.tensor([target % p == i for p in (2, 3, 5) for i in range(p)]).float()
        dataset.append(((image, concepts), target))

    return dataset


def get_data_loaders(batch_size=64):
    # Define a transform to preprocess the data (you can customize this)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the original MNIST dataset
    original_train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transform, download=True)
    original_test_dataset = datasets.MNIST(
        root='./data', train=False, transform=transform, download=True)

    # # Create dataset A
    # dataset_A_train = create_parity_dataset(original_train_dataset, {4, 5})
    # dataset_A_test = create_parity_dataset(original_test_dataset, {4, 5})

    # # Create dataset B
    # dataset_B_train = create_parity_dataset(original_train_dataset, set(range(10)) - {4, 5})
    # dataset_B_test = create_parity_dataset(original_test_dataset, set(range(10)) - {4, 5})

    # # Check the number of samples in train and test sets of each dataset
    # print(f"Dataset A - Train: {len(dataset_A_train)}, Test: {len(dataset_A_test)}")
    # print(f"Dataset B - Train: {len(dataset_B_train)}, Test: {len(dataset_B_test)}")

    # Create DataLoader instances for train and test sets of both datasets A and B
    train_dataset = create_parity_dataset(original_train_dataset, set(range(10)))
    test_dataset = create_parity_dataset(original_test_dataset, set(range(10)))
    original_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    original_test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # train_loader_A = DataLoader(dataset_A_train, batch_size=batch_size, shuffle=True)
    # test_loader_A = DataLoader(dataset_A_test, batch_size=batch_size, shuffle=False)

    # train_loader_B = DataLoader(dataset_B_train, batch_size=batch_size, shuffle=True)
    # test_loader_B = DataLoader(dataset_B_test, batch_size=batch_size, shuffle=False)

    return (
        original_train_loader, original_test_loader,
        None, None, None, None,
        # train_loader_A, test_loader_A,
        # train_loader_B, test_loader_B,
    )
