import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.utils import load_dataset, to_categorical
from torch.utils.data import DataLoader, TensorDataset

from utils.enums import Cifar100

# seeds
torch.manual_seed(0)
np.random.seed(0)


def insert_backdoor(args, x_train_party, y_train_party, example_target, backdoor, plotting = False):
    # Insert backdoor
    if plotting:
        percent_poison = args["poisoned_percent"]
    else:
        percent_poison = args.poisoned_percent

    all_indices = np.arange(len(x_train_party))
    remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]

    target_indices = list(set(all_indices) - set(remove_indices))
    num_poison = int(percent_poison * len(target_indices))
    selected_indices = np.random.choice(target_indices, num_poison, replace=False)

    poisoned_data, poisoned_labels = backdoor.poison(
        x_train_party[selected_indices], y=example_target, broadcast=True
    )

    poisoned_x_train = np.copy(x_train_party)
    poisoned_y_train = np.argmax(y_train_party, axis=1)
    for s, i in zip(selected_indices, range(len(selected_indices))):
        poisoned_x_train[s] = poisoned_data[i]
        poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))

    return poisoned_x_train, poisoned_y_train


def create_dataset_from_poisoned_data(
    args, x_train_party, y_train_party, example_target, backdoor, plotting = False
):
    poisoned_x_train, poisoned_y_train = insert_backdoor(
        args, x_train_party, y_train_party, example_target, backdoor, plotting = plotting
    )
    # poisoned_x_train_ch = np.expand_dims(poisoned_x_train, axis=1)
    poisoned_x_train_ch = np.transpose(poisoned_x_train, (0, 3, 1, 2))

    poisoned_dataset_train = TensorDataset(
        torch.Tensor(poisoned_x_train_ch), torch.Tensor(poisoned_y_train).long()
    )
    if plotting:
        poisoned_dataloader_train = DataLoader(
            poisoned_dataset_train, batch_size=args["batch_size"], shuffle=True
        )
    else:
        poisoned_dataloader_train = DataLoader(
            poisoned_dataset_train, batch_size=args.batch_size, shuffle=True
        )

    return poisoned_dataloader_train


def create_dataset_for_normal_clients(
    args, x_train_parties, y_train_parties, num_samples_per_party, plotting = False
):
    # x_train_parties_ch = np.expand_dims(x_train_parties, axis=1)
    x_train_parties_ch = np.transpose(x_train_parties, (0, 3, 1, 2))
    y_train_parties_c = np.argmax(y_train_parties, axis=1).astype(int)

    # Create PyTorch datasets for other parties
    x_train_parties = TensorDataset(
        torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long()
    )

    if plotting:
        clean_dataset_train = torch.utils.data.random_split(
            x_train_parties, [num_samples_per_party for _ in range(1, args["num_clients"])]
        )
    else:
        clean_dataset_train = torch.utils.data.random_split(
            x_train_parties, [num_samples_per_party for _ in range(1, args.num_clients)]
        )

    

    return clean_dataset_train


def load_cifar100():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(Cifar100.MEAN, Cifar100.STD),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(Cifar100.MEAN, Cifar100.STD)]
    )
    cifar100_train = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    cifar100_test = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )

    x_train = []
    y_train = []
    for i in range(len(cifar100_train)):
        data, label = cifar100_train[i]
        x_train.append(data.numpy())
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = []
    y_test = []
    for i in range(len(cifar100_test)):
        data, label = cifar100_test[i]
        x_test.append(data.numpy())
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Set channels last
    x_train = x_train.transpose((0, 2, 3, 1))
    x_test = x_test.transpose((0, 2, 3, 1))

    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)

    return (x_train, y_train), (x_test, y_test)


def load_data(dataset):
    if dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = load_cifar100()
    else:  # dataset in [mnist, cifar10]
        (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(dataset)

    # label must be one hot encoded
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    return x_train, y_train, x_test, y_test


def create_train_loaders(args, x_train, y_train, example_target, backdoor, plotting = False):
    if plotting:
        num_samples = y_train.shape[0]
        num_samples_erased_party = int(num_samples / args["num_clients"])
        num_samples_per_party = int(
            (num_samples - num_samples_erased_party) / (args["num_clients"] - 1)
        )

        num_samples = (args["num_clients"] - 1) * num_samples_per_party

        x_train_party = x_train[0:num_samples_erased_party]
        y_train_party = y_train[0:num_samples_erased_party]

        x_train_parties = x_train[
            num_samples_erased_party : num_samples_erased_party + num_samples
        ]
        y_train_parties = y_train[
            num_samples_erased_party : num_samples_erased_party + num_samples
        ]

        poisoned_dataloader_train = create_dataset_from_poisoned_data(
            args, x_train_party, y_train_party, example_target, backdoor, plotting = plotting
        )
        clean_dataset_train = create_dataset_for_normal_clients(
            args, x_train_parties, y_train_parties, num_samples_per_party, plotting = plotting
        )

        train_loaders = [poisoned_dataloader_train]
        for i in range(len(clean_dataset_train)):
            train_loaders.append(
                DataLoader(clean_dataset_train[i], batch_size=args["batch_size"], shuffle=True)
            )
            
    else:
        num_samples = y_train.shape[0]
        num_samples_erased_party = int(num_samples / args.num_clients)
        num_samples_per_party = int(
            (num_samples - num_samples_erased_party) / (args.num_clients - 1)
        )

        num_samples = (args.num_clients - 1) * num_samples_per_party

        x_train_party = x_train[0:num_samples_erased_party]
        y_train_party = y_train[0:num_samples_erased_party]

        x_train_parties = x_train[
            num_samples_erased_party : num_samples_erased_party + num_samples
        ]
        y_train_parties = y_train[
            num_samples_erased_party : num_samples_erased_party + num_samples
        ]

        poisoned_dataloader_train = create_dataset_from_poisoned_data(
            args, x_train_party, y_train_party, example_target, backdoor, plotting = plotting
        )
        clean_dataset_train = create_dataset_for_normal_clients(
            args, x_train_parties, y_train_parties, num_samples_per_party, plotting = plotting
        )

        train_loaders = [poisoned_dataloader_train]
        for i in range(len(clean_dataset_train)):
            train_loaders.append(
                DataLoader(clean_dataset_train[i], batch_size=args.batch_size, shuffle=True)
            )
    

    return train_loaders


def create_test_loaders(args, x_test, y_test, example_target, backdoor, plotting = False):
    all_indices = np.arange(len(x_test))
    remove_indices = all_indices[np.all(y_test == example_target, axis=1)]

    target_indices = list(set(all_indices) - set(remove_indices))
    poisoned_data, poisoned_labels = backdoor.poison(
        x_test[target_indices], y=example_target, broadcast=True
    )

    poisoned_x_test = np.copy(x_test)
    poisoned_y_test = np.argmax(y_test, axis=1)

    for s, i in zip(target_indices, range(len(target_indices))):
        poisoned_x_test[s] = poisoned_data[i]
        poisoned_y_test[s] = int(np.argmax(poisoned_labels[i]))

    # poisoned_x_test_ch = np.expand_dims(poisoned_x_test, axis=1)
    poisoned_x_test_ch = np.transpose(poisoned_x_test, (0, 3, 1, 2))

    poisoned_dataset_test = TensorDataset(
        torch.Tensor(poisoned_x_test_ch), torch.Tensor(poisoned_y_test).long()
    )
    testloader_poison = DataLoader(
        poisoned_dataset_test, batch_size=1000, shuffle=False
    )

    # x_test_pt = np.expand_dims(x_test, axis=1)
    x_test_pt = np.transpose(x_test, (0, 3, 1, 2))

    y_test_pt = np.argmax(y_test, axis=1).astype(int)
    dataset_test = TensorDataset(
        torch.Tensor(x_test_pt), torch.Tensor(y_test_pt).long()
    )
    testloader = DataLoader(dataset_test, batch_size=1000, shuffle=False)

    return testloader, testloader_poison


def get_loaders(args, plotting = False):
    if plotting:
        x_train, y_train, x_test, y_test = load_data(args['dataset'])
    else:
        x_train, y_train, x_test, y_test = load_data(args.dataset)

    # Init backdoor pattern
    backdoor = PoisoningAttackBackdoor(add_pattern_bd)

    example_target = np.zeros(y_train.shape[1])
    example_target[-1] = 1

    train_loaders = create_train_loaders(
        args, x_train, y_train, example_target, backdoor, plotting=plotting
    )
    test_loader, test_loader_poison = create_test_loaders(
        args, x_test, y_test, example_target, backdoor, plotting=plotting
    )

    return train_loaders, test_loader, test_loader_poison
