import torch
import torch.nn as nn   
import torch.optim as optim
import helper_utils as utils

distances = torch.tensor([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5],
    [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5],
    [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5],
    [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5], [20.0]
], dtype=torch.float32)

# Corresponding delivery times in minutes
times = torch.tensor([
    [6.96], [9.67], [12.11], [14.56], [16.77], [21.7], [26.52], [32.47], [37.15], [42.35],
    [46.1], [52.98], [57.76], [61.29], [66.15], [67.63], [69.45], [71.57], [72.8], [73.88],
    [76.34], [76.38], [78.34], [80.07], [81.86], [84.45], [83.98], [86.55], [88.33], [86.83],
    [89.24], [88.11], [88.16], [91.77], [92.27], [92.13], [90.73], [90.39], [92.98]
], dtype=torch.float32)

# First normalize the data

distances_mean = distances.mean()
distances_std = distances.std()

# Calculate the mean and standard deviation for the 'times' tensor
times_mean = times.mean()
times_std = times.std()

# Apply standardization to the distances.
distances_norm = (distances - distances_mean) / distances_std

# Apply standardization to the times.
times_norm = (times - times_mean) / times_std

# Building non-linear model with activation function
torch.manual_seed(27)
model = nn.Sequential(
    nn.Linear(1,3),
    nn.ReLU(),
    nn.Linear(3,1)
)
# training the model
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

TOTAL_EPOCHS = 3000

# Frames bunched early, where the line actually moves. Even spacing would spend
# most of the animation on a curve that has already stopped changing.
frame_epochs = utils.frame_schedule(TOTAL_EPOCHS, n_frames=80)

# One figure, created once. The loop below moves the line on it rather than
# building a new chart each time.
plot = utils.setup_live_plot(distances_norm, times_norm)

# Each recorded frame is replayed into the GIF after training finishes
frames = []

# training loop
for epoch in range(TOTAL_EPOCHS):
    optimizer.zero_grad()
    output = model(distances_norm)
    loss = loss_fn(output, times_norm)
    loss.backward()
    optimizer.step()

    # Redraw only on scheduled epochs, and keep the frame for the GIF
    if (epoch + 1) in frame_epochs:
        y_pred = utils.update_live_plot(
            plot=plot,
            epoch=epoch,
            loss=loss,
            model=model,
            distances_norm=distances_norm
        )
        frames.append((epoch + 1, loss.item(), y_pred))

print("\nTraining Complete.")
print(f"\nFinal Loss: {loss.item()}")

utils.save_training_gif(frames, distances_norm, times_norm,
                        path='Course-1/training_progress.gif')

distance_to_predict = 5.1
with torch.no_grad():
    # Normalize the input distance
    distance_tensor = torch.tensor([[distance_to_predict]], dtype=torch.float32)
    new_distance_norm = (distance_tensor - distances_mean) / distances_std
    
    # Get the normalized prediction from the model
    predicted_time_norm = model(new_distance_norm)
    
    # De-normalize the output to get the actual time in minutes
    predicted_time_actual = (predicted_time_norm * times_std) + times_mean
    
    # --- Decision Making Logic ---
    print(f"Prediction for a {distance_to_predict}-mile delivery: {predicted_time_actual.item():.1f} minutes")
    
    # First, check if the delivery is possible within the 45-minute timeframe
    if predicted_time_actual.item() > 45:
        print("\nDecision: Do NOT promise the delivery in under 45 minutes.")
    else:
        # If it is possible, then determine the vehicle based on the distance
        if distance_to_predict <= 3:
            print(f"\nDecision: Yes, delivery is possible. Since the distance is {distance_to_predict} miles (<= 3 miles), use a bike.")
        else:
            print(f"\nDecision: Yes, delivery is possible. Since the distance is {distance_to_predict} miles (> 3 miles), use a car.")