"""
[Note]
1. Running Distributed computing in jupyter notebook is tricky, so I write a sample code in python script (.py).
2. pip install numpy==1.23.5
[2](https://stackoverflow.com/questions/71689095/how-to-solve-the-pytorch-runtimeerror-numpy-is-not-available-without-upgrading)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#########################################
# Shared Model Definition (Same for both)
#########################################
class MNISTModel(nn.Module):
    def __init__(self):
        # Step 1: Initialize parent class and define layers
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, 3, 1)
        self.conv2   = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1     = nn.Linear(9216, 128)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x):
        # Step 2: Pass data through layers with activations and pooling
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#########################################
# SECTION 1: Single GPU Training
#########################################
def train_single_gpu():
    # Step 1: Set device to GPU (assuming at least one GPU is available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Step 2: Instantiate the model and move to the selected device
    model = MNISTModel().to(device)

    # Step 3: Prepare the MNIST dataset with transformations (normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Step 4: Setup optimizer and loss criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Step 5: Training loop (for 5 epochs)
    print("Starting Single GPU Training")
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 200 == 0:
                print(f'[Single GPU] Epoch [{epoch}/5], Batch [{batch_idx}/{len(loader)}], Loss: {loss.item():.4f}')
    print("Finished Single GPU Training\n")

#########################################
# SECTION 2: Distributed Data Parallel (DDP)
#########################################
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def train_ddp(rank, world_size):
    # Step 1 (DDP): Initialize the process group with NCCL backend (for GPUs)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

    # Step 2 (DDP): Set the current GPU device (each process uses one GPU)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # Step 3 (DDP): Create the model and move it to the assigned GPU, then wrap with DDP
    model = MNISTModel().to(device)
    ddp_model = DDP(model, device_ids=[rank])

    # Step 4 (DDP): Prepare MNIST dataset and use DistributedSampler to shard data among GPUs
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=False,  # Assume already downloaded in single-GPU training
        transform=transform
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    # Step 5 (DDP): Setup optimizer and loss criterion
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Step 6 (DDP): Training loop – each process of rank runs the same code on its shard of data
    if rank == 0:
        print("Starting Distributed Data Parallel (DDP) Training")
    for epoch in range(5):
        # Set epoch to shuffle data differently at each epoch
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Only one process (rank 0) prints progress to avoid multiple prints
            if rank == 0 and batch_idx % 100 == 0:
                print(f'[DDP] Epoch [{epoch}/5], Batch [{batch_idx}/{len(loader)}], Loss: {loss.item():.4f}')

    # Step 7 (DDP): Clean up by destroying the process group
    dist.destroy_process_group()
    if rank == 0:
        print("Finished Distributed Data Parallel (DDP) Training\n")

#########################################
# Main Entry: Choose a training mode to run.
#########################################
if __name__ == "__main__":
    mode = input("Run (1) Single GPU or (2) Distributed Data Parallel (DDP)? Enter 1 or 2: ").strip()
    
    if mode == "1":
        # Run single GPU training
        train_single_gpu()
    
    elif mode == "2":
        # Set environment variables needed for DDP (master address and port)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Determine the number of GPUs available (world_size)
        world_size = torch.cuda.device_count()
        if world_size < 2:
            print("Need at least 2 GPUs for DDP training. Exiting.")
        else:
            # mp.spawn will launch a process on each GPU.
            mp.set_start_method('spawn', force=True)
            mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
    
    else:
        print("Invalid selection. Please run again and choose 1 or 2.")
        
# Below is the log how it's printed (in case, you don't have multi GPU to run the test.)
'''
Run (1) Single GPU or (2) Distributed Data Parallel (DDP)? Enter 1 or 2: 1
Starting Single GPU Training

[Single GPU] Epoch [0/5], Batch [0/938], Loss: 2.3154
[Single GPU] Epoch [0/5], Batch [200/938], Loss: 0.1407
[Single GPU] Epoch [0/5], Batch [400/938], Loss: 0.1093
[Single GPU] Epoch [0/5], Batch [600/938], Loss: 0.0910
[Single GPU] Epoch [0/5], Batch [800/938], Loss: 0.0436
[Single GPU] Epoch [1/5], Batch [0/938], Loss: 0.0444
[Single GPU] Epoch [1/5], Batch [200/938], Loss: 0.1074
[Single GPU] Epoch [1/5], Batch [400/938], Loss: 0.0629
[Single GPU] Epoch [1/5], Batch [600/938], Loss: 0.0459
[Single GPU] Epoch [1/5], Batch [800/938], Loss: 0.0113
[Single GPU] Epoch [2/5], Batch [0/938], Loss: 0.0349
[Single GPU] Epoch [2/5], Batch [200/938], Loss: 0.0150
[Single GPU] Epoch [2/5], Batch [400/938], Loss: 0.0567
[Single GPU] Epoch [2/5], Batch [600/938], Loss: 0.0392
[Single GPU] Epoch [2/5], Batch [800/938], Loss: 0.0450
[Single GPU] Epoch [3/5], Batch [0/938], Loss: 0.0600
[Single GPU] Epoch [3/5], Batch [200/938], Loss: 0.0108
[Single GPU] Epoch [3/5], Batch [400/938], Loss: 0.0044
[Single GPU] Epoch [3/5], Batch [600/938], Loss: 0.0059
[Single GPU] Epoch [3/5], Batch [800/938], Loss: 0.1645
[Single GPU] Epoch [4/5], Batch [0/938], Loss: 0.0492
[Single GPU] Epoch [4/5], Batch [200/938], Loss: 0.0327
[Single GPU] Epoch [4/5], Batch [400/938], Loss: 0.0915
[Single GPU] Epoch [4/5], Batch [600/938], Loss: 0.0564
[Single GPU] Epoch [4/5], Batch [800/938], Loss: 0.0048
Finished Single GPU Training

=========================================================
Run (1) Single GPU or (2) Distributed Data Parallel (DDP)? Enter 1 or 2: 2
Starting Distributed Data Parallel (DDP) Training

[DDP] Epoch [0/5], Batch [0/118], Loss: 2.3161
[DDP] Epoch [0/5], Batch [100/118], Loss: 0.0502
[DDP] Epoch [1/5], Batch [0/118], Loss: 0.0994
[DDP] Epoch [1/5], Batch [100/118], Loss: 0.0300
[DDP] Epoch [2/5], Batch [0/118], Loss: 0.0242
[DDP] Epoch [2/5], Batch [100/118], Loss: 0.0186
[DDP] Epoch [3/5], Batch [0/118], Loss: 0.1244
[DDP] Epoch [3/5], Batch [100/118], Loss: 0.0224
[DDP] Epoch [4/5], Batch [0/118], Loss: 0.0504
[DDP] Epoch [4/5], Batch [100/118], Loss: 0.0784
Finished Distributed Data Parallel (DDP) Training
'''