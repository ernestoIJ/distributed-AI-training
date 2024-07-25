import torch
from torch import nn
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from mnist_arch_model import MNISTModel
from pathlib import Path
import time 
import os

def save_model(model, dds: bool):
   MODEL_FOLDER = Path('models')
   MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
   MODEL_FILENAME = f"model_distributed_AI_training.pt" if dds else f"model_0_distributed_AI_training.pt"
   MODEL_SAVE_PATH = MODEL_FOLDER / MODEL_FILENAME
   torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
   print(f"Saved model to {MODEL_SAVE_PATH}")


def get_data(rank, world_size, dds: bool):
   data_path = './data/MNIST/raw'
   if not os.path.exists(data_path):
      download = True
   else :
      download = False

   transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5 ))])
   train_set = MNIST(root='./data', train=True, download=download, transform=transform)

   if dds:
      train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
      train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=False, sampler=train_sampler)
   else:
      train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

   return train_loader


def get_optimizer_and_loss_fn(model, lr):
   # Loss function
   loss_fn = nn.CrossEntropyLoss()
   # optimizer 
   optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
   return loss_fn, optimizer


def dds_setup(rank, world_size):
   """
   Setup the PyTorch distributed environment.
   Args:
      rank: Unique identifer of each process
      world_size: Total number of process
   """
   os.environ["MASTER_ADDR"] = 'localhost'
   os.environ["MASTER_PORT"] = '12355'
   dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)


def dds_cleanup():
   """
   Clean up the PyTorch distributed environment.
   """
   dist.destroy_process_group()


def train_loop(train_loader, device, model, loss_fn, optimizer, rank, time_dict, epochs):
   start_time = time.time()
   for epoch in range(epochs):
      train_loss = 0
      for batch, (X_train, y_train) in enumerate(train_loader):
         X_train, y_train = X_train.to(device), y_train.to(device)
         model.train()

         # 1. forward pass
         train_pred = model.forward(X_train)

         # 2. Calculate the loss
         loss = loss_fn(train_pred, y_train)
         train_loss += loss

         # 3. zero out grad
         optimizer.zero_grad()

         # 4. backward prop
         loss.backward()

         # 5. optimizer step
         optimizer.step()

      train_loss /= len(train_loader)
      print_stmt = f"Rank: {rank} | Epoch: {epoch+1} | Train Loss: {train_loss:.4f}" if rank != None else f"Epoch: {epoch+1} | Train Loss: {train_loss:.4f}"
      print(print_stmt)
   end_time = time.time()
   name = "DDS_Training_time" if rank != None else "Without_DDS_Training_time"
   time_dict[name] = f"{end_time - start_time:.2f} seconds"
   return model


def dds_train(rank, world_size, time_dict, epochs, lr):
   """
   The core training function, where the model is trained on distributed nodes.
   Args:
      rank: Unique identifer of each process
      world_size: Total number of process
   """
   dds_setup(rank=rank, world_size=world_size)

   train_loader = get_data(rank, world_size, dds=True)

   # Model setup
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = MNISTModel(input_shape=784, output_shape=10, hidden_units=500).to(device)
   model = DDP(model)

   loss_fn, optimizer = get_optimizer_and_loss_fn(model=model, lr=lr)

   # Train loop
   epochs = epochs

   print("Starting Training with DDS")
   model = train_loop(train_loader=train_loader, device=device, model=model, loss_fn=loss_fn, optimizer=optimizer, rank=rank, time_dict=time_dict, epochs=epochs)
         
   # Cleaning the distributed environment
   dds_cleanup()
   save_model(model=model, dds=True)


def train(time_dict, epochs, lr):

   train_loader = get_data(rank=None, world_size=None, dds=False)
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = MNISTModel(input_shape=784, output_shape=10, hidden_units=500).to(device)

   loss_fn, optimizer = get_optimizer_and_loss_fn(model=model, lr=lr)

   epochs = epochs

   print("Starting Training without DDS")
   model = train_loop(train_loader=train_loader, device=device, model=model, loss_fn=loss_fn, optimizer=optimizer, rank=None, time_dict=time_dict, epochs=epochs)
   save_model(model=model, dds=False)  

def main(epochs):
   manager = mp.Manager()
   time_dict = manager.dict()
   epochs = epochs
   lr = 0.01
   world_size = 3
   mp.spawn(dds_train, args=(world_size, time_dict, epochs, lr), nprocs=world_size, join=True)
   train(time_dict, epochs=epochs, lr=lr)
   # print('\n')
   # print(time_dict)
   return time_dict