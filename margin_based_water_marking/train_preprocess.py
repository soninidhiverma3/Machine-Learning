import os
import argparse
import copy
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
import waitGPU
import wandb
from PIL import Image

# Initialize WandB
wandb.login()
CIFAR_QUERY_SIZE = (3, 32, 32)

# Set up WandB run
run = wandb.init(project="margin-project", name="Margin_project_ML")

# Define data preprocessing and transformations
def get_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    return transform

# Data visualization helper function
def visualize_batch(loader, title="Sample Data Batch"):
    """Visualize a batch of images from the data loader."""
    images, labels = next(iter(loader))
    images = make_grid(images, nrow=8, normalize=True)
    npimg = images.numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Training loop with preprocessing and visualization
def loop(model, query_model, loader, opt, lr_scheduler, epoch, logger, output_dir, max_epoch=100, train_type='standard', mode='train', device='cuda', addvar=None):
    meters = MultiAverageMeter(['nat loss', 'nat acc', 'query loss', 'query acc'])

    for batch_idx, batch in enumerate(loader):
        images = batch[0]
        labels = batch[1].long()
        epoch_with_batch = epoch + (batch_idx+1) / len(loader)
        
        # Apply learning rate scheduler
        if lr_scheduler is not None:
            lr_new = lr_scheduler(epoch_with_batch)
            for param_group in opt.param_groups:
                param_group.update(lr=lr_new)

        images, labels = images.to(device), labels.to(device)
        if mode == 'train':
            model.train()
            opt.zero_grad()

        preds = model(images)
        nat_acc = (preds.topk(1, dim=1).indices == labels.unsqueeze(1)).all(1).float().mean()
        nat_loss = F.cross_entropy(preds, labels, reduction='none')

        if train_type == 'none':
            with torch.no_grad():
                model.eval()
                query, response = query_model()
                query_preds = model(query)
                query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
                query_loss = F.cross_entropy(query_preds, response)
                if mode == 'train':
                    model.train()
            loss = nat_loss.mean()
        elif train_type == 'base':
            query, response = query_model()
            query_preds = model(query)
            query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
            query_loss = F.cross_entropy(query_preds, response, reduction='none')
            loss = torch.cat([nat_loss, query_loss]).mean()
        elif train_type == 'margin':
            num_sample = int(np.interp([epoch], [0, max_epoch], [25, 25])[0])
            if mode == 'train':
                query, response = query_model(discretize=False, num_sample=num_sample)
                for _ in range(5):
                    query = query.detach().requires_grad_(True)
                    query_preds = model(query)
                    query_loss = F.cross_entropy(query_preds, response)
                    query_loss.backward()
                    query = query + query.grad.sign() * (1/255)
                    query = query_model.project(query)
                    model.zero_grad()
            else:
                query, response = query_model(discretize=(mode!='train'))
            query_preds = model(query)
            query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
            query_loss = addvar * F.cross_entropy(query_preds, response, reduction='none')
            loss = torch.cat([nat_loss, query_loss]).mean()
            wandb.log({"query_acc": query_acc, "loss": loss})

        if mode == 'train':
            loss.backward()
            opt.step()

        # Update and log metrics
        meters.update({
            'nat loss': nat_loss.mean().item(),
            'nat acc': nat_acc.item(),
            'query loss': query_loss.mean().item(),
            'query acc': query_acc.item()
        }, n=images.size(0))

        if batch_idx % 100 == 0 and mode == 'train':
            logger.info('=====> {} {}'.format(mode, str(meters)))

    logger.info("({:3.1f}%) Epoch {:3d} - {} {}".format(100*epoch/max_epoch, epoch, mode.capitalize().ljust(6), str(meters)))
    
    if mode == 'test' and (epoch+1) % 20 == 0:
        save_image(query.cpu(), os.path.join(output_dir, "images", f"query_image_{epoch}.png"), nrow=query.size(0))
        image_path = os.path.join(output_dir, "images", f"query_image_{epoch}.png")
        wandb.log({"query_image_epoch": wandb.Image(image_path, caption=f"Query Image at Epoch {epoch}")})
    return meters

def train(args, output_dir):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'output.log')),
            logging.StreamHandler()
        ])

    # Dataset loading with transformations
    transform = get_transforms()
    if args.dataset == 'cifar10':
        query_size = CIFAR_QUERY_SIZE
        model_archive = cifar10.models
        train_loader, valid_loader, test_loader = get_cifar10_loaders(transform)
    elif args.dataset == 'cifar100':
        query_size = CIFAR_QUERY_SIZE
        model_archive = cifar10.models_cifar100
        train_loader, valid_loader, test_loader = get_cifar100_loaders(transform)
    elif args.dataset == 'svhn':
        query_size = CIFAR_QUERY_SIZE
        model_archive = resnet.models
        train_loader, valid_loader, test_loader = get_svhn_loaders(transform)

    response_scale = 100 if args.dataset == 'cifar100' else 10
    model = model_archive[args.model_type](num_classes=response_scale)
    query = queries.queries[args.query_type](query_size=(args.num_query, *query_size),
                                response_size=(args.num_query,), query_scale=255, response_scale=response_scale)

    # Visualize data samples
    visualize_batch(train_loader, title="Training Data Sample")
    wandb.log({"train_batch_samples": wandb.Image(make_grid(next(iter(train_loader))[0], nrow=8))})

    model.to(args.device)
    query.to(args.device)

    # Optimizer and scheduler setup
    opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
    lr_scheduler = lambda t: np.interp([t], [0, 100, 100, 150, 150, 200], [0.1, 0.1, 0.01, 0.01, 0.001, 0.001])[0]

    for epoch in range(args.epoch):
        train_meters = loop(model, query, train_loader, opt, lr_scheduler, epoch, logger, output_dir,
                            train_type=args.train_type, max_epoch=args.epoch, mode='train', device=args.device, addvar=args.variable)
        with torch.no_grad():
            val_meters = loop(model, query, valid_loader, opt, lr_scheduler, epoch, logger, output_dir,
                              train_type=args.train_type, max_epoch=args.epoch, mode='val', device=args.device, addvar=args.variable)
            test_meters = loop(model, query, test_loader, opt, lr_scheduler, epoch, logger, output_dir,
                               train_type=args.train_type, max_epoch=args.epoch, mode='test', device=args.device, addvar=args.variable)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training with data preprocessing and visualization')
    parser.add_argument("--dir", type=str, default='experiments')
    parser.add_argument("--dataset", type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument("--train-type", type=str, default='margin', choices=['none', 'base', 'margin'])
    parser.add_argument("--model-type", type=str, default='res34', choices=['res18', 'res34', 'res50'])
    parser.add_argument("--query-type", type=str, default='stochastic', choices=['stochastic'])
    parser.add_argument("--num-query", type=int, default=10)
    parser.add_argument("--num-mixup", type=int, default=1)
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
