import torch
import torch.nn as nn   
import torch.optim as optim
import helper_utils as utils

# In this lab, you'll build and train your first neural network, a single neuron that learns patterns from data to make predictions.

# A bike delivery person with a 7-mile delivery order. Your company promises delivery in under 30 minutes, and one more late delivery could put your job at risk. 
# Can you make this delivery on time? Your neural network will learn from historical delivery data to help you decide.

# Following the Machine Learning (ML) pipeline from the lecture videos, you will:

# Prepare delivery data, the distances and times from past orders.
# Build a simple neural network using PyTorch (just one neuron!).
# Train it to find the relationship between distance and delivery time.
# Predict whether you can make that 7-mile delivery in time.

torch.manual_seed(42)  # Set the random seed for reproducibility

# Data Preparation: Define the distances and times for past deliveries

# Stage 1 & 2: Data Ingestion and Preparation

# Time to prepare your delivery data for training. In the ML pipeline, this combines two stages: Data Ingestion (gathering raw data) 
# and Data Preparation (cleaning it up). In more realistic projects, you'd pull delivery logs from a data source and fix errors or missing values. 
# For this lab, that work is already done, but with a twist. This isn't the same data from the lecture videos. You're looking at a different set of deliveries, 
# which means your model might find a different pattern and make a different prediction for that 7-mile delivery.
# Define the two essential tensors for your task:
# The distances tensor contains how far you biked for four recent deliveries (in miles).
# The times tensor shows how long each delivery took (in minutes).
# dtype=torch.float32 sets your data type to 32-bit floating point values for precise calculations.

distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)


# Model Building: Create a simple neural network with one neuron
# Stage 3: Model Building

# Now you'll create your model (this is stage 3 of the ML pipeline: Model Building). For bike deliveries, you'll assume a linear relationship between distance and
# time, a reasonable starting point. Your model will be a single neuron that learns this relationship.

# Remember from the lecture videos, a single neuron with one input implements a linear equation:

# Time = W × Distance + B

# Your job is to find the best values for the weight (W) and bias (B) that fit your delivery data.

# Use nn.Sequential(nn.Linear(1, 1)) to create a linear model.
# nn.Linear(1, 1): The first 1 means it takes one input (distance), and the second 1 means one neuron that is producing one output (predicted time).
# This single linear layer will automatically manage the weight and bias parameters for you.

model = nn.Sequential(nn.Linear(1, 1))
loss_fn = nn.MSELoss()  # Mean Squared Error loss function for regression tasks
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer with a learning rate of 0.01

# Training the Model: Fit the model to your delivery data
# Stage 4: Model Training
for epoch in range(500):  # Train for 1000 epochs
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear previous gradients
    predictions = model(distances)  # Forward pass: compute predicted times
    loss = loss_fn(predictions, times)  # Compute the loss
    loss.backward()  # Backward pass: compute gradients
    optimizer.step()  # Update weights and biases
    if (epoch + 1) % 50 == 0:  # Print loss every 100 epochs
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
        
# utils.plot_results(model, distances, times)

# Make a Prediction: Use the trained model to predict delivery time for a 7-mile order
with torch.no_grad():
    new_distance = torch.tensor([[7.0]], dtype=torch.float32)
    predicted_time = model(new_distance)
    print(f'Predicted delivery time for a 7-mile order: {predicted_time.item():.2f} minutes')
    
#  Model's Learning

weight, bias = model[0].weight.item(), model[0].bias.item()
print(f'Learned weight (W): {weight:.4f}, Learned bias (B): {bias:.4f}')


# Now we are going to test the model with new data that is non-linear. This will help us see how well our linear model generalizes to different types of data.
new_distances = torch.tensor([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5],
    [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5],
    [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5],
    [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5], [20.0]
], dtype=torch.float32)

# Corresponding delivery times in minutes
new_times = torch.tensor([
    [6.96], [9.67], [12.11], [14.56], [16.77], [21.7], [26.52], [32.47], [37.15], [42.35],
    [46.1], [52.98], [57.76], [61.29], [66.15], [67.63], [69.45], [71.57], [72.8], [73.88],
    [76.34], [76.38], [78.34], [80.07], [81.86], [84.45], [83.98], [86.55], [88.33], [86.83],
    [89.24], [88.11], [88.16], [91.77], [92.27], [92.13], [90.73], [90.39], [92.98]
], dtype=torch.float32)

# Visualize the new data
utils.plot_nonlinear_comparison(model, new_distances, new_times)

with torch.no_grad():
    predictions = model(new_distances)
    new_loss = loss_fn(predictions, new_times)
    print(f'Loss on new non-linear data: {new_loss.item():.4f}')