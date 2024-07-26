# **Distributed Learning Demonstrator**

The Distributed Learning Demonstrator is an interactive platform designed to showcase the effectiveness 	
of distributed machine learning techniques compared to traditional non-distributed methods. By utilizing 		
the MNIST datasets, this project allows users to train a simple neural network model using both distributed 	
and non-distributed approaches directly through a web interface. This tool not only serves educational purposes by helping users understand and visualize the impact of 		
distributed training on model performance and training speed, but it also provides practical experience with setting up and managing distributed training environments. 
Users can adjust parameters like the number of epochs and learning rates, and then observe how these changes affect the training outcomes in real-time.
The project is built on PyTorch and leverages Flask for the web backend, offering a user-friendly interface to initiate training sessions, visualize results, 
and compare metrics such as accuracy, precision, recall, and F1-score between the two training methodologies. This platform is ideal for students, educators, 
and machine learning enthusiasts looking to deepen their understanding of distributed computing's role in AI and machine learning.

## **Setting it up**

Start the Backend Server: Open a terminal window and navigate to the `backend` directory of the project by executing
```bash
cd backend
```
Once in the `backend` directory, start the Flask server by running:
```bash
python app.py
```
Start the Frontend interface: Open a terminal window and navigate to the `frontend` directory of the project by executing
```bash
cd frontend
```
In the `frontend` directory, start the HTTP server by running:
```bash
python -m http.server
```

## **MNIST Model Architecture**
The MNISTModel is a neural network built with PyTorch, designed specifically for recognizing handwritten digits from the MNIST dataset. It features two fully connected layers: the first layer transforms the input into a hidden layer with a configurable number of units, applying a ReLU activation for non-linearity. The output layer maps the activated features to ten output classes corresponding to the digits 0 through 9. This model structure provides a straightforward framework for demonstrating fundamental neural network operations within educational or experimental settings.
```bash
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
  def __init__(self, input_shape: int, output_shape: int, hidden_units: int):
    super(MNISTModel, self).__init__()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(in_features=input_shape, out_features=hidden_units)
    self.fc2 = nn.Linear(in_features=hidden_units, out_features=output_shape)

  def forward(self, x):
    x = self.flatten(x)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
```
