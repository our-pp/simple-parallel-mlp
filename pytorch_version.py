import statistics
from argparse import ArgumentParser
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms

losses: list[float] = []

class Net(nn.Module):
    def __init__(self, hidden_dim: int):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = torch.sigmoid(self.fc2(x))
        return x


@dataclass
class CLIArgs:
    batch_size: int
    hidden_size: int
    epoch_count: int
    force_cpu: bool


def configure_args() -> CLIArgs:
    parser = ArgumentParser()
    parser.add_argument("--batch-size", "-b", default=1024, type=int, dest="batch_size")
    parser.add_argument(
        "--hidden-size", "-s", default=300, type=int, dest="hidden_size"
    )
    parser.add_argument("--epoch-count", "-e", default=10, type=int, dest="epoch_count")
    parser.add_argument("--cpu", action="store_true", dest="force_cpu")
    return parser.parse_args()


def get_device(force_cpu: bool = False):
    if not torch.cuda.is_available() or force_cpu:
        return torch.device("cpu")
    return torch.device("cuda")


def load_dataset(batch_size: int):
    train_dataset = datasets.MNIST(
        "./dataset/",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    validation_dataset = datasets.MNIST(
        "./dataset/", train=False, transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False
    )

    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, validation_loader


def train(
    epoch: int,
    log_interval: int = 200,
    *,
    model: Net,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device = get_device(),
    optimizer: torch.optim.Optimizer,
    criterion: Callable[..., Any] = nn.MSELoss(),
):
    # Set model to training mode
    model.train()
    total_loss = 0

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)

        # Calculate loss
        target_onehot = torch.zeros([len(data), 10]).to(device)
        target_onehot = target_onehot.scatter(1, target.unsqueeze(-1), 1)

        loss = criterion(output, target_onehot)

        # Backpropagate
        loss.backward()
        total_loss += loss.data.item()

        # Update weights
        optimizer.step()

    print(
        f"Train Epoch: {epoch} "
        f"Loss: {total_loss / len(train_loader):.6f}"
    )
    losses.append(total_loss / len(train_loader))


def validate(
    loss_vector: list[float],
    accuracy_vector: list[float],
    *,
    model: Net,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device = get_device(),
    criterion: Callable[..., Any] = nn.MSELoss(),
    validation_loader: torch.utils.data.DataLoader,
):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)

        target_onehot = torch.zeros([len(data), 10]).to(device)
        target_onehot = target_onehot.scatter(1, target.unsqueeze(-1), 1)

        val_loss += criterion(output, target_onehot).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100.0 * correct.to(torch.float32) / len(train_loader.dataset)
    accuracy_vector.append(accuracy)

    print(
        f"\nValidation set: Average loss: {val_loss:.4f}, "
        f"Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.0f}%)\n"
    )


def main() -> None:
    args = configure_args()

    device = get_device(args.force_cpu)

    train_loader, validation_loader = load_dataset(args.batch_size)

    model = Net(hidden_dim=args.hidden_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)

    lossv, accv = [], []
    times: list[float] = []
    for epoch in range(1, args.epoch_count + 1):
        start = perf_counter()
        train(epoch, device=get_device(args.force_cpu), log_interval=1, model=model, train_loader=train_loader, optimizer=optimizer)
        end = perf_counter()
        times.append(end - start)
        # validate(
        #     lossv,
        #     accv,
        #     model=model,
        #     train_loader=train_loader,
        #     validation_loader=validation_loader,
        # )

    print(statistics.mean(times))

    x = range(1, args.epoch_count + 1)
    plt.plot(x, losses)
    plt.title("PyTorch Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("image/pytorch-loss.png")


if __name__ == "__main__":
    main()
