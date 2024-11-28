import os
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from models import cifar10, resnet, queries
from loaders import get_cifar10_loaders, get_cifar100_loaders, get_svhn_loaders
from utils import MultiAverageMeter
import wandb

# Initialize WandB
wandb.login()
CIFAR_QUERY_SIZE = (3, 32, 32)

# WandB setup
run = wandb.init(project="margin-project", name="Margin_project_ML")

# Define data preprocessing and transformations
def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

# Visualize a batch of images and save the plot
def visualize_batch(loader, title="Sample Data Batch", save_path=None):
    """Visualize a batch of images from the data loader."""
    images, labels = next(iter(loader))
    images = make_grid(images, nrow=8, normalize=True)
    npimg = images.numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)  # Save plot
    plt.show()

# Function to plot and save training metrics
def plot_metrics(metrics, output_dir, metric_name="Accuracy"):
    """Plot and save training and validation metrics."""
    plt.figure()
    for phase in metrics:
        plt.plot(metrics[phase], label=f"{phase} {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f"{metric_name} over Epochs")
    plot_path = os.path.join(output_dir, f"{metric_name.lower()}_plot.png")
    plt.savefig(plot_path)  # Save the plot
    plt.close()
    return plot_path

# Training loop
def loop(model, query_model, loader, opt, lr_scheduler, epoch, logger, output_dir, 
         max_epoch=100, train_type='standard', mode='train', device='cuda', addvar=None):
    meters = MultiAverageMeter(['nat loss', 'nat acc', 'query loss', 'query acc'])
    
    for batch_idx, batch in enumerate(loader):
        images, labels = batch[0].to(device), batch[1].long().to(device)
        epoch_with_batch = epoch + (batch_idx + 1) / len(loader)
        
        # Learning rate adjustment
        if lr_scheduler is not None:
            lr_new = lr_scheduler(epoch_with_batch)
            for param_group in opt.param_groups:
                param_group.update(lr=lr_new)

        if mode == 'train':
            model.train()
            opt.zero_grad()

        preds = model(images)
        nat_acc = (preds.argmax(dim=1) == labels).float().mean()
        nat_loss = F.cross_entropy(preds, labels, reduction='mean')

        # Calculate query loss and accuracy based on train type
        if train_type == 'margin':
            query, response = query_model()
            query_preds = model(query)
            query_acc = (query_preds.argmax(dim=1) == response).float().mean()
            query_loss = F.cross_entropy(query_preds, response, reduction='mean')
            loss = nat_loss + addvar * query_loss
        else:
            loss = nat_loss
            query_loss, query_acc = torch.tensor(0.0), torch.tensor(0.0)

        # Backpropagation and optimizer step
        if mode == 'train':
            loss.backward()
            opt.step()
        
        # Update metrics
        meters.update({
            'nat loss': nat_loss.item(),
            'nat acc': nat_acc.item(),
            'query loss': query_loss.item(),
            'query acc': query_acc.item()
        }, n=images.size(0))

        # Logging metrics to WandB
        wandb.log({
            "epoch": epoch,
            "nat_loss": nat_loss.item(),
            "nat_acc": nat_acc.item(),
            "query_loss": query_loss.item(),
            "query_acc": query_acc.item()
        })

    # Logging to console and WandB
    logger.info(f"{mode.capitalize()} Epoch [{epoch}/{max_epoch}]: {meters}")
    return meters

# Training function with metrics tracking and plotting
def train(args, output_dir):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'output.log')),
            logging.StreamHandler()
        ]
    )

    # Load data
    transform = get_transforms()
    if args.dataset == 'cifar10':
        train_loader, valid_loader, test_loader = get_cifar10_loaders(transform)
    elif args.dataset == 'cifar100':
        train_loader, valid_loader, test_loader = get_cifar100_loaders(transform)
    elif args.dataset == 'svhn':
        train_loader, valid_loader, test_loader = get_svhn_loaders(transform)

    # Model and optimizer setup
    response_scale = 100 if args.dataset == 'cifar100' else 10
    model = cifar10.models[args.model_type](num_classes=response_scale).to(args.device)
    query = queries.queries[args.query_type]((args.num_query, *CIFAR_QUERY_SIZE), response_scale=response_scale).to(args.device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
    lr_scheduler = lambda t: np.interp([t], [0, 100, 100, 150, 150, 200], [0.1, 0.1, 0.01, 0.01, 0.001, 0.001])[0]

    # Visualize and save a batch of training data
    visualize_batch(train_loader, title="Training Data Sample", save_path=os.path.join(output_dir, "sample_batch.png"))

    # Initialize tracking metrics
    training_metrics = {"train acc": [], "train loss": [], "val acc": [], "val loss": []}

    for epoch in range(args.epoch):
        # Training and validation loops
        train_meters = loop(model, query, train_loader, opt, lr_scheduler, epoch, logger, output_dir, 
                            train_type=args.train_type, max_epoch=args.epoch, mode='train', device=args.device, addvar=args.variable)
        
        with torch.no_grad():
            val_meters = loop(model, query, valid_loader, opt, lr_scheduler, epoch, logger, output_dir,
                              train_type=args.train_type, max_epoch=args.epoch, mode='val', device=args.device, addvar=args.variable)

        # Update metrics for plotting
        training_metrics["train acc"].append(train_meters['nat acc'])
        training_metrics["train loss"].append(train_meters['nat loss'])
        training_metrics["val acc"].append(val_meters['nat acc'])
        training_metrics["val loss"].append(val_meters['nat loss'])

    # Save and log plots for accuracy and loss
    acc_plot_path = plot_metrics({"Train": training_metrics["train acc"], "Validation": training_metrics["val acc"]}, 
                                 output_dir, metric_name="Accuracy")
    loss_plot_path = plot_metrics({"Train": training_metrics["train loss"], "Validation": training_metrics["val loss"]}, 
                                  output_dir, metric_name="Loss")
    wandb.log({"accuracy_plot": wandb.Image(acc_plot_path), "loss_plot": wandb.Image(loss_plot_path)})

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training with data preprocessing and visualization')
    parser.add_argument("--dir", type=str, default='experiments')
    parser.add_argument("--dataset", type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument("--train-type", type=str, default='margin', choices=['none', 'base', 'margin'])
    parser.add_argument("--model-type", type=str, default='res34', choices=['res18', 'res34', 'res50'])
    parser.add_argument("--query-type", type=str, default='stochastic', choices=['stochastic'])
    parser.add_argument("--num-query", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--variable", type=float, default=0.1)
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    exp_name = f"{args.dataset}_{args.model_type}_{args.train_type}_{args.num_query}"
    output_dir = os.path.join(args.dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    train(args, output_dir)
import os
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from models import cifar10, resnet, queries
from loaders import get_cifar10_loaders, get_cifar100_loaders, get_svhn_loaders
from utils import MultiAverageMeter
import wandb

# Initialize WandB
wandb.login()
CIFAR_QUERY_SIZE = (3, 32, 32)

# WandB setup
run = wandb.init(project="margin-project", name="Margin_project_ML")

# Define data preprocessing and transformations
def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

# Visualize a batch of images and save the plot
def visualize_batch(loader, title="Sample Data Batch", save_path=None):
    """Visualize a batch of images from the data loader."""
    images, labels = next(iter(loader))
    images = make_grid(images, nrow=8, normalize=True)
    npimg = images.numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)  # Save plot
    plt.show()

# Function to plot and save training metrics
def plot_metrics(metrics, output_dir, metric_name="Accuracy"):
    """Plot and save training and validation metrics."""
    plt.figure()
    for phase in metrics:
        plt.plot(metrics[phase], label=f"{phase} {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f"{metric_name} over Epochs")
    plot_path = os.path.join(output_dir, f"{metric_name.lower()}_plot.png")
    plt.savefig(plot_path)  # Save the plot
    plt.close()
    return plot_path

# Training loop
def loop(model, query_model, loader, opt, lr_scheduler, epoch, logger, output_dir, 
         max_epoch=100, train_type='standard', mode='train', device='cuda', addvar=None):
    meters = MultiAverageMeter(['nat loss', 'nat acc', 'query loss', 'query acc'])
    
    for batch_idx, batch in enumerate(loader):
        images, labels = batch[0].to(device), batch[1].long().to(device)
        epoch_with_batch = epoch + (batch_idx + 1) / len(loader)
        
        # Learning rate adjustment
        if lr_scheduler is not None:
            lr_new = lr_scheduler(epoch_with_batch)
            for param_group in opt.param_groups:
                param_group.update(lr=lr_new)

        if mode == 'train':
            model.train()
            opt.zero_grad()

        preds = model(images)
        nat_acc = (preds.argmax(dim=1) == labels).float().mean()
        nat_loss = F.cross_entropy(preds, labels, reduction='mean')

        # Calculate query loss and accuracy based on train type
        if train_type == 'margin':
            query, response = query_model()
            query_preds = model(query)
            query_acc = (query_preds.argmax(dim=1) == response).float().mean()
            query_loss = F.cross_entropy(query_preds, response, reduction='mean')
            loss = nat_loss + addvar * query_loss
        else:
            loss = nat_loss
            query_loss, query_acc = torch.tensor(0.0), torch.tensor(0.0)

        # Backpropagation and optimizer step
        if mode == 'train':
            loss.backward()
            opt.step()
        
        # Update metrics
        meters.update({
            'nat loss': nat_loss.item(),
            'nat acc': nat_acc.item(),
            'query loss': query_loss.item(),
            'query acc': query_acc.item()
        }, n=images.size(0))

        # Logging metrics to WandB
        wandb.log({
            "epoch": epoch,
            "nat_loss": nat_loss.item(),
            "nat_acc": nat_acc.item(),
            "query_loss": query_loss.item(),
            "query_acc": query_acc.item()
        })

    # Logging to console and WandB
    logger.info(f"{mode.capitalize()} Epoch [{epoch}/{max_epoch}]: {meters}")
    return meters

# Training function with metrics tracking and plotting
def train(args, output_dir):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'output.log')),
            logging.StreamHandler()
        ]
    )

    # Load data
    transform = get_transforms()
    if args.dataset == 'cifar10':
        train_loader, valid_loader, test_loader = get_cifar10_loaders(transform)
    elif args.dataset == 'cifar100':
        train_loader, valid_loader, test_loader = get_cifar100_loaders(transform)
    elif args.dataset == 'svhn':
        train_loader, valid_loader, test_loader = get_svhn_loaders(transform)

    # Model and optimizer setup
    response_scale = 100 if args.dataset == 'cifar100' else 10
    model = cifar10.models[args.model_type](num_classes=response_scale).to(args.device)
    query = queries.queries[args.query_type]((args.num_query, *CIFAR_QUERY_SIZE), response_scale=response_scale).to(args.device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
    lr_scheduler = lambda t: np.interp([t], [0, 100, 100, 150, 150, 200], [0.1, 0.1, 0.01, 0.01, 0.001, 0.001])[0]

    # Visualize and save a batch of training data
    visualize_batch(train_loader, title="Training Data Sample", save_path=os.path.join(output_dir, "sample_batch.png"))

    # Initialize tracking metrics
    training_metrics = {"train acc": [], "train loss": [], "val acc": [], "val loss": []}

    for epoch in range(args.epoch):
        # Training and validation loops
        train_meters = loop(model, query, train_loader, opt, lr_scheduler, epoch, logger, output_dir, 
                            train_type=args.train_type, max_epoch=args.epoch, mode='train', device=args.device, addvar=args.variable)
        
        with torch.no_grad():
            val_meters = loop(model, query, valid_loader, opt, lr_scheduler, epoch, logger, output_dir,
                              train_type=args.train_type, max_epoch=args.epoch, mode='val', device=args.device, addvar=args.variable)

        # Update metrics for plotting
        training_metrics["train acc"].append(train_meters['nat acc'])
        training_metrics["train loss"].append(train_meters['nat loss'])
        training_metrics["val acc"].append(val_meters['nat acc'])
        training_metrics["val loss"].append(val_meters['nat loss'])

    # Save and log plots for accuracy and loss
    acc_plot_path = plot_metrics({"Train": training_metrics["train acc"], "Validation": training_metrics["val acc"]}, 
                                 output_dir, metric_name="Accuracy")
    loss_plot_path = plot_metrics({"Train": training_metrics["train loss"], "Validation": training_metrics["val loss"]}, 
                                  output_dir, metric_name="Loss")
    wandb.log({"accuracy_plot": wandb.Image(acc_plot_path), "loss_plot": wandb.Image(loss_plot_path)})

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training with data preprocessing and visualization')
    parser.add_argument("--dir", type=str, default='experiments')
    parser.add_argument("--dataset", type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument("--train-type", type=str, default='margin', choices=['none', 'base', 'margin'])
    parser.add_argument("--model-type", type=str, default='res34', choices=['res18', 'res34', 'res50'])
    parser.add_argument("--query-type", type=str, default='stochastic', choices=['stochastic'])
    parser.add_argument("--num-query", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--variable", type=float, default=0.1)
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    exp_name = f"{args.dataset}_{args.model_type}_{args.train_type}_{args.num_query}"
    output_dir = os.path.join(args.dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    train(args, output_dir)
