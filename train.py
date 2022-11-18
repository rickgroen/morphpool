import torch
from tqdm import tqdm

# project imports.
from utils.options import get_args
from loaders import create_loader
from networks import create_network


def train(args):
    device = torch.device(args.device)
    # Init dataset and data loader
    train_loader, test_loader, n_classes = create_loader(args.dataset, args.batch_size)

    # Initialize the networks.
    network = create_network(args, n_classes)
    network = network.to(device)

    # Set the optimizer and loss function.
    optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    # The learning rate should end at about 2% of the initial learning rate
    lr_gamma = -2 ** (-1 / args.epochs) * -5 ** (-2 / args.epochs)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Record which inputs should be used.
    use_depth = args.use_depth

    # Loop for the amount of specified epochs.
    print("Starting training...")
    for epoch in range(args.epochs):
        network.train()
        train_loss = 0.0
        tbar = tqdm(train_loader)
        print(f"[epoch {epoch}] Learning rate is {optimizer.param_groups[0]['lr']:.4f}.")
        for idx, (image, depth, semantic) in enumerate(tbar):
            optimizer.zero_grad(set_to_none=True)
            if use_depth:
                prediction = network(depth.to(device))
            else:
                prediction = network(image.to(device))
            semantic = semantic.to(device)
            loss = loss_fn(prediction.squeeze(1), semantic.squeeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            tbar.set_description(f'({epoch}) Train loss: {train_loss / (idx + 1):.6f}')
        # Update learning rate.
        lr_scheduler.step()
    # Naming and saving after training.
    network.save_model(args.dataset)


if __name__ == '__main__':
    args = get_args()
    print(f"Running code using CUDA {torch.version.cuda} on device {args.device}")
    train(args)

    print("YOU ARE TERMINATED!")
