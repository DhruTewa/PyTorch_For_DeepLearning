import time
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

# Animation pacing, tune these two to taste
FRAME_PAUSE = 0.15   # seconds the live window rests on each frame
GIF_FPS = 6          # frames per second in the saved GIF



def plot_data(distances, times, normalize=False):
    """
    Creates a scatter plot of the data points.

    Args:
        distances: The input data points for the x-axis.
        times: The target data points for the y-axis.
        normalize: A boolean flag indicating whether the data is normalized.
    """
    # Create a new figure with a specified size
    plt.figure(figsize=(8, 6))

    # Plot the data points as a scatter plot
    plt.plot(distances.numpy(), times.numpy(), color='orange', marker='o', linestyle='none', label='Actual Delivery Times')

    # Check if the data is normalized to set appropriate labels and title
    if normalize:
        # Set the plot title for normalized data
        plt.title('Normalized Delivery Data (Bikes & Cars)')
        # Set the x-axis label for normalized data
        plt.xlabel('Normalized Distance')
        # Set the y-axis label for normalized data
        plt.ylabel('Normalized Time')
        # Display the legend
        plt.legend()
        # Add a grid to the plot
        plt.grid(True)
        # Show the plot
        plt.show()

    # Handle the case for un-normalized data
    else:
        # Set the plot title for un-normalized data
        plt.title('Delivery Data (Bikes & Cars)')
        # Set the x-axis label for un-normalized data
        plt.xlabel('Distance (miles)')
        # Set the y-axis label for un-normalized data
        plt.ylabel('Time (minutes)')
        # Display the legend
        plt.legend()
        # Add a grid to the plot
        plt.grid(True)
        # Show the plot
        plt.show()
    
    

def plot_final_fit(model, distances, times, distances_norm, times_std, times_mean):
    """
    Plots the predictions of a trained model against the original data,
    after de-normalizing the predictions.

    Args:
        model: The trained model used for prediction.
        distances: The original, un-normalized input data.
        times: The original, un-normalized target data.
        distances_norm: The normalized input data for the model.
        times_std: The standard deviation used for de-normalization.
        times_mean: The mean value used for de-normalization.
    """
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations for prediction
    with torch.no_grad():
        # Get predictions from the model using normalized data
        predicted_norm = model(distances_norm)

    # De-normalize the predictions to their original scale
    predicted_times = (predicted_norm * times_std) + times_mean

    # Create a new figure for the plot
    plt.figure(figsize=(8, 6))

    # Plot the original data points
    plt.plot(distances.numpy(), times.numpy(), color='orange', marker='o', linestyle='none', label='Actual Data (Bikes & Cars)')

    # Plot the de-normalized predictions from the model
    plt.plot(distances.numpy(), predicted_times.numpy(), color='green', label='Non-Linear Model Predictions')

    # Set the title of the plot
    plt.title('Non-Linear Model Fit vs. Actual Data')
    # Set the x-axis label
    plt.xlabel('Distance (miles)')
    # Set the y-axis label
    plt.ylabel('Time (minutes)')
    # Add a legend to the plot
    plt.legend()
    # Enable the grid
    plt.grid(True)
    # Display the plot
    plt.show()

    

def plot_training_progress(epoch, loss, model, distances_norm, times_norm):
    """
    Plots the training progress of a model on normalized data,
    showing the current fit at each epoch.

    Args:
        epoch: The current training epoch number.
        loss: The loss value at the current epoch.
        model: The model being trained.
        distances_norm: The normalized input data.
        times_norm: The normalized target data.
    """
    # Clear the previous plot from the output cell
    # clear_output(wait=True)

    # Make predictions using the current state of the model
    predicted_norm = model(distances_norm)

    # Convert tensors to NumPy arrays for plotting
    x_plot = distances_norm.numpy()
    y_plot = times_norm.numpy()
    
    # Detach predictions from the computation graph and convert to NumPy
    y_pred_plot = predicted_norm.detach().numpy()

    # Sort the data based on distance to ensure a smooth line plot
    sorted_indices = x_plot.argsort(axis=0).flatten()

    # Create a new figure for the plot
    plt.figure(figsize=(8, 6))

    # Plot the original normalized data points
    plt.plot(x_plot, y_plot, color='orange', marker='o', linestyle='none', label='Actual Normalized Data')

    # Plot the model's predictions as a line
    plt.plot(x_plot[sorted_indices], y_pred_plot[sorted_indices], color='green', label='Model Predictions')

    # Set the title of the plot, including the current epoch
    plt.title(f'Epoch: {epoch + 1} | Normalized Training Progress')
    # Set the x-axis label
    plt.xlabel('Normalized Distance')
    # Set the y-axis label
    plt.ylabel('Normalized Time')
    # Display the legend
    plt.legend()
    # Add a grid to the plot
    plt.grid(True)
    # Show the plot
    plt.show()

    # Pause briefly to allow the plot to be rendered
    time.sleep(0.05)


# ---------------------------------------------------------------------------
# Animated training progress
#
# The old plot_training_progress above builds a NEW figure every call and then
# blocks on plt.show(). These functions instead build ONE figure up front and
# move the existing line on each update, which is what produces motion.
# ---------------------------------------------------------------------------


@dataclass
class LivePlot:
    """Handles for the single reused figure, passed back in on every update."""

    fig: Figure
    ax: Axes
    line: Line2D
    x: np.ndarray       # flattened inputs
    order: np.ndarray   # indices that sort x left to right


def frame_schedule(total_epochs: int, n_frames: int = 80) -> set[int]:
    """
    Which epochs to capture as animation frames.

    Spacing them evenly wastes most of the animation: the fit is ~80% settled
    within the first few hundred epochs, so evenly spaced frames show a snap
    followed by a long stretch of nothing moving. Log spacing packs frames in
    early, where the line is actually changing, and thins them out later.

    Returns fewer than n_frames, since early values collide after rounding.
    """
    spaced = np.logspace(0, np.log10(total_epochs), n_frames)
    return set(np.unique(spaced.astype(int)).tolist())


def _build_figure(x: np.ndarray, y: np.ndarray) -> LivePlot:
    """
    Creates the figure shared by both the live view and the saved GIF.

    The green prediction line starts empty. Later calls move it with
    set_data rather than drawing a new one.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, y, color='orange', marker='o', linestyle='none',
            label='Actual Normalized Data')
    (line,) = ax.plot([], [], color='green', label='Model Predictions')

    # Fixed limits, otherwise the axes rescale each frame and the plot jitters
    ax.set_xlim(x.min() - 0.2, x.max() + 0.2)
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax.set_xlabel('Normalized Distance')
    ax.set_ylabel('Normalized Time')
    ax.grid(True)
    ax.legend()

    return LivePlot(fig=fig, ax=ax, line=line, x=x, order=x.argsort())


def _predict(model: torch.nn.Module, distances_norm: torch.Tensor) -> np.ndarray:
    """Returns the model's current predictions as a flat NumPy array."""
    # We only want the numbers to draw, so skip building the gradient graph
    with torch.no_grad():
        predicted = model(distances_norm)
    return predicted.numpy().flatten()


def setup_live_plot(distances_norm: torch.Tensor, times_norm: torch.Tensor) -> LivePlot:
    """Call once BEFORE the training loop. Opens the window that will animate."""
    # Interactive mode, so show() returns immediately instead of blocking the loop
    plt.ion()

    plot = _build_figure(distances_norm.numpy().flatten(),
                         times_norm.numpy().flatten())
    plot.fig.show()
    return plot


def update_live_plot(plot: LivePlot, epoch: int, loss: torch.Tensor,
                     model: torch.nn.Module, distances_norm: torch.Tensor) -> np.ndarray:
    """
    Call inside the training loop. Moves the green line to the current fit.

    Returns the predictions so the caller can record them for the GIF.
    """
    y_pred = _predict(model, distances_norm)

    # Move the existing line rather than drawing a new one, sorted left to right
    plot.line.set_data(plot.x[plot.order], y_pred[plot.order])

    plot.ax.set_title(f'Epoch: {epoch + 1} | Loss: {loss.item():.4f}')

    # Push the change to the screen, then rest so the eye can follow it
    plot.fig.canvas.draw_idle()
    plot.fig.canvas.flush_events()

    # Guard the zero case: plt.pause(0) waits on the event loop forever
    if FRAME_PAUSE > 0:
        plt.pause(FRAME_PAUSE)

    return y_pred


def save_training_gif(frames: list[tuple[int, float, np.ndarray]],
                      distances_norm: torch.Tensor, times_norm: torch.Tensor,
                      path: str = 'training_progress.gif') -> None:
    """
    Call once AFTER training. Replays the recorded frames into a GIF.

    Args:
        frames: list of (epoch, loss value, predictions) captured during training.
        path: where to write the .gif file.
    """
    plot = _build_figure(distances_norm.numpy().flatten(),
                         times_norm.numpy().flatten())

    def draw_frame(i: int) -> tuple[Line2D]:
        """Draws frame i. FuncAnimation calls this once per frame."""
        epoch, loss_value, y_pred = frames[i]
        plot.line.set_data(plot.x[plot.order], y_pred[plot.order])
        plot.ax.set_title(f'Epoch: {epoch} | Loss: {loss_value:.4f}')
        return (plot.line,)

    animation = FuncAnimation(plot.fig, draw_frame, frames=len(frames), blit=False)
    animation.save(path, writer='pillow', fps=GIF_FPS)

    plt.close(plot.fig)
    print(f"Saved animation to {path}")
    
    
def plot_delivery_data(df):
    """
    Generates a scatter plot of delivery data.

    This function plots delivery time vs. distance, with markers colored by the
    time of day and styled (filled/hollow) based on whether the delivery
    was on a weekend.

    Args:
        df: A DataFrame containing the required columns:
            'distance_miles', 'delivery_time_minutes',
            'time_of_day_hours', and 'is_weekend'.
    """
    # Use a style to get a base grid for the plot
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create the plot figure and axes
    plt.figure(figsize=(12, 7))
    ax = plt.axes()

    # Define colors for the graph's dark theme
    background_color = '#35466A'
    # Color for most text elements
    text_color = 'black'
    grid_color = '#FFFFFF'
    # Shade of grey for legend markers to pop on background
    legend_marker_color = '#CCCCCC'

    # Set the dark background color for the plot area
    ax.set_facecolor(background_color)

    # Separate data for weekend and weekday deliveries
    df_weekend = df[df['is_weekend'] == 1]
    df_weekday = df[df['is_weekend'] == 0]

    # Create a normalizer for consistent color mapping across time of day
    # Choose a colormap for time of day
    cmap = plt.get_cmap('YlOrRd')
    norm = Normalize(vmin=df['time_of_day_hours'].min(), vmax=df['time_of_day_hours'].max())

    # Plot the weekday data (filled circles)
    # Circles are filled and colored by time of day
    scatter_weekday = plt.scatter(
        df_weekday['distance_miles'],
        df_weekday['delivery_time_minutes'],
        s=80,
        # Color based on time of day
        c=df_weekday['time_of_day_hours'],
        cmap=cmap,
        norm=norm,
        alpha=0.9,
        # Label for automatic legend entry
        label='Weekday'
    )

    # Plot the weekend data (hollow circles)
    # Circles are hollow with edges colored by time of day
    scatter_weekend = plt.scatter(
        df_weekend['distance_miles'],
        df_weekend['delivery_time_minutes'],
        s=80,
        # Makes circles hollow
        facecolors='none',
        # Edge colored by time of day
        edgecolors=cmap(norm(df_weekend['time_of_day_hours'])),
        linewidths=1.5,
        alpha=0.9,
        # Label for automatic legend entry
        label='Weekend'
    )

    # Style all non-legend text elements and plot lines
    title_font = {'family': 'sans-serif', 'color': text_color, 'weight': 'bold', 'size': 16}
    label_font = {'family': 'sans-serif', 'color': text_color, 'weight': 'normal', 'size': 12}

    # Main plot titles and axis labels
    ax.set_title('Delivery Time vs. Distance', fontdict=title_font)
    ax.set_xlabel('Distance (miles)', fontdict=label_font)
    ax.set_ylabel('Delivery Time (minutes)', fontdict=label_font)

    # Axis tick parameters and spine (border) colors
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)

    # Grid lines styling
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=grid_color, alpha=0.5)

    # Create custom legend handles to ensure generic gray markers
    # Filled gray circle for Weekday in legend
    weekday_handle = Line2D([0], [0], marker='o', color='w', label='Weekday',
                            markerfacecolor=legend_marker_color, markersize=8,
                            markeredgecolor=legend_marker_color, markeredgewidth=1.5,
                            # Prevents a line from being drawn through the marker
                            linestyle='None')

    # Hollow gray circle for Weekend in legend
    weekend_handle = Line2D([0], [0], marker='o', color='w', label='Weekend',
                            markerfacecolor='none', markersize=8,
                            markeredgecolor=legend_marker_color, markeredgewidth=1.5,
                            # Prevents a line from being drawn through the marker
                            linestyle='None')

    # Add and Style Legend using the custom handles and text colors
    # Explicitly provide handles and labels to override default scatter legend entries
    legend = ax.legend(handles=[weekday_handle, weekend_handle],
                       labels=['Weekday', 'Weekend'],
                       title='Day Type')

    # Set the color for all legend text items (labels) to white
    plt.setp(legend.get_texts(), color='white', fontsize=12)
    # Set the color for the legend title to white
    plt.setp(legend.get_title(), color='white', fontsize=13, weight='bold')

    # Add and style the colorbar for Time of Day
    # The colorbar uses the 'scatter_weekday' plot for its color mapping source
    cbar = plt.colorbar(scatter_weekday, ax=ax)
    cbar.set_label('Time of Day (Hour)', fontdict=label_font)
    # Colorbar tick labels also in black
    cbar.ax.tick_params(colors=text_color)

    # Display the plot
    plt.show()


    
def plot_rush_hour(data_df, features):
    """
    Generates a scatter plot of delivery data, grouped by rush hour.
    Features larger markers and custom legend text for better visibility on a dark background.

    Args:
        data_df: DataFrame with original data ('distance_miles', 'delivery_time_minutes').
        features: The prepared features tensor with the 'is_rush_hour' column.
    """
    # Set the plotting style to include a grid
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create the figure and axes for the plot
    plt.figure(figsize=(12, 7))
    ax = plt.axes()

    # Define colors for the plot's visual theme
    # Background color of the plot area
    background_color = '#35466A'
    # Color for "Rush Hour" data points (orange)
    rush_hour_color = '#E64C29'
    # Color for "Not Rush Hour" data points (light grey)
    default_color = '#F0F0E0'
    # General color for axis labels, tick labels, and main titles
    text_color = 'black'
    # Color for grid lines
    grid_color = '#FFFFFF'

    # Apply the defined background color to the plot axes
    ax.set_facecolor(background_color)

    # Prepare the 'is_rush_hour' boolean array from features for data separation
    is_rush_hour = features[:, 3].numpy().astype(bool)

    # Plot data points for "Rush Hour" deliveries
    # Data points are large, filled, and colored with 'rush_hour_color'.
    # Their edge color now matches their fill color for consistency.
    scatter_rush_hour = plt.scatter(
        data_df['distance_miles'][is_rush_hour],
        data_df['delivery_time_minutes'][is_rush_hour],
        # Marker size for visibility
        s=90,
        c=rush_hour_color,
        # Opacity for slight transparency
        alpha=0.9,
        # Edge color matches fill color
        edgecolor=rush_hour_color,
        # Line width for marker edges
        linewidth=0.5,
        # Label for legend entry
        label='Rush Hour'
    )

    # Plot data points for "Not Rush Hour" deliveries
    # Data points are large, filled, and colored with 'default_color'.
    scatter_not_rush_hour = plt.scatter(
        data_df['distance_miles'][~is_rush_hour],
        data_df['delivery_time_minutes'][~is_rush_hour],
        # Marker size for visibility
        s=90,
        c=default_color,
        # Opacity for slight transparency
        alpha=0.7,
        # Label for legend entry
        label='Not Rush Hour'
    )

    # Define font properties for plot titles and labels
    title_font = {'family': 'sans-serif', 'color': text_color, 'weight': 'bold', 'size': 16}
    label_font = {'family': 'sans-serif', 'color': text_color, 'weight': 'normal', 'size': 12}

    # Set the main plot title and axis labels
    ax.set_title('Delivery Time vs. Distance (Grouped by Delivery Time Slots)', fontdict=title_font)
    ax.set_xlabel('Distance (miles)', fontdict=label_font)
    ax.set_ylabel('Delivery Time (minutes)', fontdict=label_font)

    # Style axis ticks and the plot's outer border (spines)
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)

    # Add and style grid lines for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=grid_color, alpha=0.5)

    # Custom Legend Configuration
    # Create Line2D objects to serve as custom legend handles.
    # These handles display the desired marker style and color without a connecting line.

    # Handle for "Rush Hour" (filled orange circle)
    rush_hour_handle = Line2D([0], [0], marker='o', color='w', label='Rush Hour',
                              markerfacecolor=rush_hour_color, markersize=10,
                              # Edge color matches fill color
                              markeredgecolor=rush_hour_color,
                              markeredgewidth=0.5, linestyle='None')

    # Handle for "Not Rush Hour" (filled light grey circle)
    not_rush_hour_handle = Line2D([0], [0], marker='o', color='w', label='Not Rush Hour',
                                  markerfacecolor=default_color, markersize=10,
                                  markeredgecolor='none', linestyle='None')

    # Create the legend using the custom handles and labels, with a specific title
    legend = ax.legend(handles=[rush_hour_handle, not_rush_hour_handle],
                       labels=['Rush Hour', 'Not Rush Hour'],
                       title='Delivery Time Slot')

    # Set the background and edge color for the legend frame
    legend.get_frame().set_facecolor('#4A6C8C')
    legend.get_frame().set_edgecolor('none')

    # Style the legend title: set color to white and adjust font size/weight/alignment
    plt.setp(legend.get_title(), color='white', fontsize=15, weight='bold',
             # Horizontal alignment for the title
             ha='left')

    # Style the legend labels: set their color to match their data points and adjust font size
    text_labels = legend.get_texts()
    text_labels[0].set_color(rush_hour_color)
    text_labels[0].set_fontsize(14)
    text_labels[1].set_color(default_color)
    text_labels[1].set_fontsize(14)

    # Display the plot
    plt.show()


    
def plot_final_data(features, targets):
    """
    Generates a scatter plot of the final, processed data,
    ensuring all four possible categories appear in the legend.

    Args:
        features: The prepared 4D features tensor.
        targets: The prepared targets tensor.
    """
    # Set the plotting style to include a grid
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create the figure and axes for the plot
    plt.figure(figsize=(12, 7))
    ax = plt.axes()

    # Define colors from the theme
    background_color = '#35466A'
    # Color for rush hour data points
    rush_hour_color = '#E64C29'
    # Color for non-rush hour/default data points
    default_color = '#F0F0E0'
    # General color for most graph text
    text_color = 'black'
    # Color for grid lines
    grid_color = '#FFFFFF'

    # Apply the defined background color to the plot axes
    ax.set_facecolor(background_color)

    # Prepare data for plotting
    normalized_distance = features[:, 0].numpy()
    delivery_time = targets.numpy()
    is_weekend = features[:, 2].numpy().astype(bool)
    is_rush_hour = features[:, 3].numpy().astype(bool)

    # Create boolean masks for each of the four delivery categories
    weekday_rush = is_rush_hour & ~is_weekend
    weekday_non_rush = ~is_rush_hour & ~is_weekend
    weekend_rush = is_rush_hour & is_weekend
    weekend_non_rush = ~is_rush_hour & is_weekend

    # Plot the four data categories
    # Plotting order here affects layer visibility for overlapping points.

    # 1. Weekday (Rush Hour): Filled orange circles
    plt.scatter(
        normalized_distance[weekday_rush],
        delivery_time[weekday_rush],
        s=90, c=rush_hour_color, alpha=0.9,
        label='Weekday (Rush Hour)'
    )

    # 2. Weekday (Not Rush Hour): Filled white/light gray circles
    plt.scatter(
        normalized_distance[weekday_non_rush],
        delivery_time[weekday_non_rush],
        s=90, c=default_color, alpha=0.7,
        label='Weekday (Not Rush Hour)'
    )

    # 3. Weekend (Rush Hour): Hollow orange circles
    plt.scatter(
        normalized_distance[weekend_rush],
        delivery_time[weekend_rush],
        s=90, facecolors='none', edgecolors=rush_hour_color,
        linewidths=1.5, alpha=0.9,
        label='Weekend (Rush Hour)'
    )

    # 4. Weekend (Not Rush Hour): Hollow white/light gray circles
    plt.scatter(
        normalized_distance[weekend_non_rush],
        delivery_time[weekend_non_rush],
        s=90, facecolors='none', edgecolors=default_color,
        linewidths=1.5, alpha=0.7,
        label='Weekend (Not Rush Hour)'
    )

    # Style plot titles and axis labels
    title_font = {'family': 'sans-serif', 'color': text_color, 'weight': 'bold', 'size': 16}
    label_font = {'family': 'sans-serif', 'color': text_color, 'weight': 'normal', 'size': 12}

    ax.set_title('Final Processed Data for Training', fontdict=title_font)
    ax.set_xlabel('Normalized Distance (miles)', fontdict=label_font)
    ax.set_ylabel('Delivery Time (minutes)', fontdict=label_font)

    # Style axis ticks and plot border (spines)
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)

    # Add and style grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=grid_color, alpha=0.5)

    # Custom Legend Configuration
    # Create Line2D objects to serve as custom legend handles.
    # These handles will display the specified marker style and color without a connecting line.

    # Handle for Weekday (Rush Hour) - Filled Orange circle
    weekday_rush_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=rush_hour_color,
                                 markersize=10, linestyle='None',
                                 markeredgecolor=rush_hour_color, markeredgewidth=0.5)

    # Handle for Weekday (Not Rush Hour) - Filled White circle
    weekday_non_rush_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=default_color,
                                     markersize=10, linestyle='None')

    # Handle for Weekend (Rush Hour) - Hollow Orange circle
    weekend_rush_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                 markeredgecolor=rush_hour_color, markeredgewidth=1.5,
                                 markersize=10, linestyle='None')

    # Handle for Weekend (Not Rush Hour) - Hollow White circle
    weekend_non_rush_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                     markeredgecolor=default_color, markeredgewidth=1.5,
                                     markersize=10, linestyle='None')

    # Create the legend using the custom handles in the specified order and set its title
    legend = ax.legend(
        handles=[
            weekday_rush_handle,
            weekday_non_rush_handle,
            weekend_rush_handle,
            weekend_non_rush_handle
        ],
        labels=[
            'Weekday (Rush Hour)',
            'Weekday (Not Rush Hour)',
            'Weekend (Rush Hour)',
            'Weekend (Not Rush Hour)'
        ],
        title='Delivery Type'
    )

    # Style the legend frame's background and edge
    legend.get_frame().set_facecolor('#4A6C8C')
    legend.get_frame().set_edgecolor('none')

    # Style the legend title: set color to white and adjust font size/weight
    plt.setp(legend.get_title(), color='white', fontsize=15, weight='bold')

    # Style the legend labels: set their color to match respective circle colors and adjust font size
    text_labels = legend.get_texts()
    for label_text in text_labels:
        label_text.set_fontsize(14)

    text_labels[0].set_color(rush_hour_color)
    text_labels[1].set_color(default_color)
    text_labels[2].set_color(rush_hour_color)
    text_labels[3].set_color(default_color)

    # Display the plot
    plt.show()


    
def plot_model_predictions(predicted_outputs, actual_targets):
    """
    Creates a themed scatter plot to compare actual vs. predicted values.

    Args:
        predicted_outputs: The tensor of predictions from the model.
        actual_targets: The tensor of actual target values.
    """
    # Define colors from the established theme
    background_color = '#35466A'
    text_color = 'black'
    grid_color = '#FFFFFF'
    model_predictions_color = '#F0F0E0'
    perfect_prediction_line_color = '#E64C29'

    # Set the plotting style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Detach tensors and convert to NumPy arrays for plotting
    predicted_values = predicted_outputs.detach().numpy()
    actual_values = actual_targets.detach().numpy()

    # Create the plot figure and axes
    plt.figure(figsize=(12, 7))
    ax = plt.axes()
    ax.set_facecolor(background_color)

    # Create the scatter plot
    ax.scatter(
        actual_values,
        predicted_values,
        s=90,
        c=model_predictions_color,
        alpha=0.7,
        edgecolor=model_predictions_color,
        linewidth=0.5,
        label='Model Predictions'
    )

    # Add the "perfect prediction" line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(
        lims, lims,
        linestyle='-',
        color=perfect_prediction_line_color,
        alpha=0.75,
        zorder=0,
        label='Perfect Prediction'
    )

    # Define font properties
    title_font = {'family': 'sans-serif', 'color': text_color, 'weight': 'bold', 'size': 16}
    label_font = {'family': 'sans-serif', 'color': text_color, 'weight': 'normal', 'size': 12}

    # Set titles and labels
    ax.set_title('Actual vs. Predicted Delivery Times', fontdict=title_font)
    ax.set_xlabel('Actual Delivery Time (minutes)', fontdict=label_font)
    ax.set_ylabel('Predicted Delivery Time (minutes)', fontdict=label_font)

    # Style ticks and spines
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)

    # Style grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=grid_color, alpha=0.5)

    # Custom Legend Configuration
    model_pred_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=model_predictions_color,
                               markersize=10, linestyle='None')
    perfect_pred_handle = Line2D([0], [0], color=perfect_prediction_line_color, linewidth=2,
                                 label='Perfect Prediction')

    legend = ax.legend(
        handles=[model_pred_handle, perfect_pred_handle],
        labels=['Model Predictions', 'Perfect Prediction'],
        title='Prediction Insights'
    )

    # Style the legend
    legend.get_frame().set_facecolor('#4A6C8C')
    legend.get_frame().set_edgecolor('none')
    plt.setp(legend.get_title(), color='white', fontsize=15, weight='bold', ha='left')
    text_labels = legend.get_texts()
    text_labels[0].set_color(model_predictions_color)
    text_labels[0].set_fontsize(14)
    text_labels[1].set_color(perfect_prediction_line_color)
    text_labels[1].set_fontsize(14)

    # Display the plot
    plt.show()


    
def prediction(model, data_df, raw_inputs, rush_hour_feature_func):
    """
    Takes raw inputs as a tensor, validates them, prepares them for the model, 
    predicts the delivery time, and displays the results in a formatted table.

    Args:
        model: The trained PyTorch model.
        data_df: The original training DataFrame for calculating stats.
        raw_inputs: A tensor containing the raw input values
                    in the format [distance, time_of_day, is_weekend].
        rush_hour_feature_func: The function to engineer the rush hour feature.

    Returns:
        None: This function prints output directly and returns None on success.
              On validation failure, it prints an error and returns early.
    """
    # 1. Validate User Inputs
    # Extract scalar values from the input tensor for validation checks.
    distance_value = raw_inputs[0, 0].item()
    time_value = raw_inputs[0, 1].item()
    weekend_value = raw_inputs[0, 2].item()

    # Check for a valid distance.
    if distance_value <= 0:
        print(f"\033[91mError: Invalid distance value - The value must be greater than 0 and no more than 20 miles.")
        return

    # Check for a valid time.
    if time_value <= 0 or time_value > 24:
        print(f"\033[91mError: Invalid time value - The value must be greater than 0 and no more than 24.0.")
        return

    # Check if the weekend flag is a valid boolean (0 or 1).
    if weekend_value not in [0.0, 1.0]:
        print(f"\033[91mError: Invalid weekend value - Please use 1 (True) for a weekend or 0 (False) for a weekday.")
        return

    # Check if the distance exceeds the service area.
    if distance_value > 20:
        print(f"\033[91mDelivery Area Exceeded: The requested distance of {distance_value:.1f} miles exceeds the 20-mile limit.")
        return

    # Check if the time is within operating hours.
    if time_value <= 8 or time_value > 20:
        print(f"\033[91mOutside Operating Hours: Deliveries are only processed between 8:00 and 20:00.")
        return

    # 2. Recalculate Normalization Stats
    # In a real application, these stats would be saved after training and loaded here.
    dist_mean = data_df['distance_miles'].mean()
    dist_std = data_df['distance_miles'].std()
    hours_mean = data_df['time_of_day_hours'].mean()
    hours_std = data_df['time_of_day_hours'].std()

    # 3. Prepare Input Tensors for the Model
    # Slice the input tensor to get individual feature tensors.
    distance_tensor = raw_inputs[:, 0]
    hours_tensor = raw_inputs[:, 1]
    weekend_tensor = raw_inputs[:, 2]

    # Engineer the rush hour feature using the provided function.
    is_rush_hour_tensor = rush_hour_feature_func(hours_tensor, weekend_tensor)

    # Normalize the continuous features using the stats from the training data.
    normalized_distance = (distance_tensor - dist_mean) / dist_std
    normalized_hours = (hours_tensor - hours_mean) / hours_std

    # Combine all engineered and normalized features into a final tensor for the model.
    new_features = torch.cat([
        normalized_distance.unsqueeze(0),
        normalized_hours.unsqueeze(0),
        weekend_tensor.unsqueeze(0),
        is_rush_hour_tensor.unsqueeze(0)
    ], dim=1)

    # 4. Make the Prediction
    # Use torch.no_grad() for inference as we are not training the model.
    with torch.no_grad():
        # Set the model to evaluation mode.
        model.eval()
        predicted_time_tensor = model(new_features)

    # 5. Format and Display the Results
    # Extract the final prediction and format all values for display.
    predicted_time = predicted_time_tensor.item()
    time_of_week = "Weekend" if weekend_value else "Weekday"
    is_rush_hour_text = "Yes" if is_rush_hour_tensor.item() == 1.0 else "No"
    hour = int(time_value)
    minutes = int((time_value % 1) * 60)
    formatted_time = f"{hour:02d}:{minutes:02d}"

    # Define the table structure for clean output.
    header = "+{:-<42}+{:-<23}+".format('', '')
    title_line = "|{:^66}|".format(" Model Prediction ")
    line1 = f"| {'Time of the Week':<40} | {time_of_week:<21} |"
    line2 = f"| {'Distance':<40} | {f'{distance_value:.1f} miles':<21} |"
    line3 = f"| {'Time':<40} | {formatted_time:<21} |"
    line4 = f"| {'Is this considered a rush hour period?':<40} | {is_rush_hour_text:<21} |"
    final_result = f"| {'Estimated Delivery Time':<40} | {f'{predicted_time:.2f} minutes':<21} |"
    
    # Print the formatted table.
    print(header)
    print(title_line)
    print(header)
    print(line1)
    print(line2)
    print(line3)
    print(line4)
    print(header)
    print(final_result)
    print(header)

import random
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image



def display_image(image, label, title, num_ticks=6, show_values=True):
    """
    Displays an image with its corresponding label and title.

    This function handles different image formats (PIL Image and PyTorch Tensor),
    normalizes the display range, and optionally overlays pixel values on the image.

    Args:
        image: The image data to be displayed. Can be a PIL Image or a PyTorch Tensor.
        label: The label associated with the image.
        title: The title for the plot.
        num_ticks (int, optional): The number of ticks to display on the color bar. Defaults to 6.
        show_values (bool, optional): If True, overlays the numerical value of each pixel on the image. Defaults to True.
    """
    # Initialize variables for value range and image data.
    vmin_val, vmax_val = None, None
    image_data = None

    # Check if the input is a PIL Image.
    if isinstance(image, Image.Image):
        # Set the value range for a standard 8-bit image.
        vmin_val = 0
        vmax_val = 255
        # Convert the PIL Image to a NumPy array.
        image_data = np.array(image)
    # Check if the input is a PyTorch Tensor.
    elif isinstance(image, torch.Tensor):
        # Convert the tensor to a NumPy array and remove any single-dimensional entries.
        image_np = image.numpy().squeeze()
        # Determine the min and max values from the tensor for normalization.
        vmin_val = image_np.min()
        vmax_val = image_np.max()
        # Assign the NumPy array to image_data.
        image_data = image_np
    # Handle unsupported image types.
    else:
        print("Warning: Unsupported image type.")
        return

    # Create a new figure for the plot.
    plt.figure(figsize=(9, 9))
    # Display the image data as a grayscale image.
    plt.imshow(image_data, cmap='gray', vmin=vmin_val, vmax=vmax_val)
    # Set the title of the plot with the provided title and label.
    plt.title(f'{title} | Label: {label}')

    # Check if pixel values should be displayed on the image.
    if show_values:
        # Calculate a threshold to determine the color of the text (black or white).
        threshold = (vmin_val + vmax_val) / 2.0
        # Get the dimensions of the image.
        height, width = image_data.shape
        
        # Iterate over each pixel to display its value.
        for y in range(height):
            for x in range(width):
                # Get the pixel value.
                value = image_data[y, x]
                # Set text color based on the pixel's brightness.
                text_color = "white" if value < threshold else "black"
                
                # Format the text to display, handling integers and floats differently.
                text_to_display = f"{value:.0f}" if isinstance(value, np.integer) else f"{value:.1f}"
                
                # Add the pixel value as text to the plot.
                plt.text(x, y, text_to_display, 
                         ha="center", va="center", color=text_color, fontsize=6)

    # Add a grid to the plot.
    plt.grid(True, color='red', alpha=0.3, zorder=2)
    # Set the x-axis ticks.
    plt.xticks(np.arange(0, 28, 4))
    # Set the y-axis ticks.
    plt.yticks(np.arange(0, 28, 4))
    
    # Add a color bar to the plot.
    cbar = plt.colorbar()
    # Create evenly spaced ticks for the color bar.
    ticks = np.linspace(vmin_val, vmax_val, num=num_ticks)
    # Set the ticks on the color bar.
    cbar.set_ticks(ticks)
    # Format the tick labels on the color bar.
    cbar.ax.set_yticklabels([f'{t:.2f}' for t in ticks])

    # Show the final plot.
    plt.show()
    
    
    
def display_predictions(model, test_loader, device):
    """
    Displays a grid of predictions for one random sample from each class.

    Args:
        model: The trained PyTorch model.
        test_loader: The DataLoader for the test set.
        device: The device (e.g., 'cuda' or 'cpu') to run inference on.
    """
    # Ensures the model is on the specified device and in evaluation mode.
    model.to(device)
    model.eval()

    # Creates a dictionary to store indices for each class.
    class_indices = {i: [] for i in range(10)}
    
    # Populates the dictionary with the indices of all samples for each class.
    for idx, (_, label) in enumerate(test_loader.dataset):
        class_indices[label].append(idx)
        
    # Selects one random index from the list of indices for each class.
    random_indices = [random.choice(indices) for indices in class_indices.values()]
    
    # Retrieves the images and corresponding labels using the randomly selected indices.
    sample_images = torch.stack([test_loader.dataset[i][0] for i in random_indices])
    sample_labels = [test_loader.dataset[i][1] for i in random_indices]

    # Temporarily disables gradient calculation for inference.
    with torch.no_grad():
        # Passes the selected images through the model to get outputs.
        outputs = model(sample_images.to(device))
        # Gets the predicted class for each image.
        _, predictions = torch.max(outputs, 1)

    # Creates a figure and a grid of subplots for displaying the images.
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    # Sets a main title for the entire figure.
    fig.suptitle('Model Predictions for a Sample of Each Class', fontsize=16)

    # Iterates through the subplots to display each image and its prediction.
    for i, ax in enumerate(axes.flat):
        # Extracts and prepares the image, true label, and predicted label for display.
        image = sample_images[i].cpu().squeeze()
        true_label = sample_labels[i]
        predicted_label = predictions[i].item()

        # Displays the image on the current subplot.
        ax.imshow(image, cmap='gray')
        
        # Sets the title of the subplot, with color indicating if the prediction is correct.
        title_color = 'green' if true_label == predicted_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {predicted_label}", color=title_color)
        
        # Hides the axes for a cleaner visual.
        ax.axis('off')

    # Adjusts the layout to prevent titles and labels from overlapping.
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Adjusts the vertical spacing between subplots.
    plt.subplots_adjust(hspace=0.3)
    
    # Displays the plot.
    plt.show()
    
    
    
def plot_metrics(train_loss, test_acc):
    """
    Displays side-by-side plots for training loss and test accuracy over epochs.

    Args:
        train_loss (list): A list of floating-point numbers representing the
                           average training loss for each epoch.
        test_acc (list): A list of floating-point numbers representing the
                         test accuracy for each epoch.
    """
    # Get the number of epochs from the length of the loss list
    num_epochs = len(train_loss)
    # Create a 1-based epoch range for the x-axis
    epochs = range(1, num_epochs + 1)

    # Create a figure and a set of subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Training Loss ---
    ax1.plot(epochs, train_loss, marker='o', linestyle='-', color='royalblue')
    ax1.set_title('Training Loss Over Epochs', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True)
    # Ensure the x-axis ticks are integers
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Plot 2: Test Accuracy ---
    ax2.plot(epochs, test_acc, marker='o', linestyle='-', color='red')
    ax2.set_title('Test Accuracy Over Epochs', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True)
    # Ensure the x-axis ticks are integers
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust layout to prevent overlap and display the plots
    plt.tight_layout()
    plt.show()
    
    
import os
import sys

import matplotlib.pyplot as plt
from directory_tree import DisplayTree
from fastai.vision.all import show_image, show_titled_image
from tqdm.auto import tqdm


def get_dataloader_bar(dataloader, color="green"):
    """
    Creates and configures a progress bar for a given dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader providing the dataset.
        color (str): The color of the progress bar.

    Returns:
        pbar (tqdm.tqdm): The configured progress bar object.
    """
    # Get the total number of samples from the dataloader's dataset.
    num_samples = len(dataloader.dataset)

    # Initialize a tqdm progress bar with specified settings.
    pbar = tqdm(
        # Set the total number of iterations for the bar.
        total=num_samples,
        # Dynamically calculate the width of the progress bar.
        ncols=int(num_samples / 10) + 300,
        # Define the format string for the progress bar's appearance.
        bar_format="{desc} {bar} {postfix}",
        # Direct the progress bar output to the standard output stream.
        file=sys.stdout,
        # Set the color of the progress bar.
        colour=color,
    )

    # Return the configured progress bar object.
    return pbar



def update_dataloader_bar(p_bar, batch, current_bs, n_samples):
    """
    Updates the existing progress bar with the current batch information.

    Args:
        p_bar (tqdm.tqdm): The progress bar object to update.
        batch (int): The current batch index.
        current_bs (int): The size of the current batch.
        n_samples (int): The total number of samples in the dataset.
    """
    # Advance the progress bar by the number of items in the current batch.
    p_bar.update(current_bs)
    # Set the description to show the current batch number.
    p_bar.set_description(f"Batch {batch+1}")

    # Check if the current batch is the last one.
    if (batch + 1) * current_bs > n_samples:
        # Update the postfix to show the total number of samples processed.
        p_bar.set_postfix_str(f"{n_samples} of a total of  {n_samples} samples")
    else:
        # Update the postfix to show the cumulative number of samples processed.
        p_bar.set_postfix_str(
            f"{current_bs*(batch+1)} of a total of  {n_samples} samples"
        )



def get_grid(num_rows, num_cols, figsize=(16, 8)):
    """
    Creates a grid of subplots and ensures the axes object is consistently formatted.

    Args:
        num_rows (int): The number of rows in the subplot grid.
        num_cols (int): The number of columns in the subplot grid.
        figsize (tuple): The dimensions of the overall figure.

    Returns:
        fig (matplotlib.figure.Figure): The generated matplotlib figure object.
        axes (list): The formatted axes grid, structured as an iterable or 2D list.
    """
    # Create a figure and a set of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Handle the case where there is only one row.
    if num_rows == 1:
        # Ensure the axes object is iterable for consistency.
        axes = [axes]
    # Handle the case where there is only one column.
    elif num_cols == 1:
        # Ensure the axes object is a 2D list for consistent indexing.
        axes = [[ax] for ax in axes]
        
    # Return the figure and the formatted axes grid.
    return fig, axes



def print_data_folder_structure(root_dir, max_depth=1):
    """
    Prints the directory tree structure for a given root directory.

    Args:
        root_dir (str): The starting path for the directory tree.
        max_depth (int): The maximum depth for the tree traversal.
    """
    # Define the configuration settings for displaying the directory tree.
    config_tree = {
        # Specify the starting path for the directory tree.
        "dirPath": root_dir,
        # Set to False to include both files and directories.
        "onlyDirs": False,
        # Set the maximum depth for the tree traversal.
        "maxDepth": max_depth,
        # Specify a sorting option (100 typically means no specific sort).
        "sortBy": 100,
    }
    # Create and display the tree structure using the unpacked configuration.
    DisplayTree(**config_tree)


def explore_extensions(root_dir):
    """
    Explores a directory tree to group all file paths by their file extensions.

    Args:
        root_dir (str): The starting directory path to search.

    Returns:
        extensions (dict): A dictionary mapping file extensions to a list of matching file paths.
    """
    # Initialize a dictionary to store file paths, grouped by extension.
    extensions = {}
    # Walk through the directory tree starting from the root directory.
    for dirpath, _, filenames in os.walk(root_dir):
        # Iterate over each file in the current directory.
        for filename in filenames:
            # Extract the file extension and convert it to lowercase.
            ext = os.path.splitext(filename)[1].lower()
            # If the extension has not been seen before, add it to the dictionary.
            if ext not in extensions:
                # Initialize a new list for this extension.
                extensions[ext] = []
            # Append the full path of the file to the list for its extension.
            extensions[ext].append(os.path.join(dirpath, filename))
    # Return the dictionary of extensions and their corresponding file paths.
    return extensions



def quick_debug(img):
    """
    Prints basic debugging information about an image tensor.

    Args:
        img (torch.Tensor): The image tensor to inspect.
    """
    # Print the shape of the image tensor.
    print(f"Shape: {img.shape}")  # Should be [3, 224, 224]
    # Print the data type of the tensor.
    print(f"Type: {img.dtype}")  # Should be torch.float32
    # Print the minimum and maximum pixel values in the tensor.
    print(
        f"Range of pixel values: [{img.min():.1f}, {img.max():.1f}]"
    )
    
    
import os

import matplotlib.pyplot as plt
import numpy as np
from directory_tree import DisplayTree
from fastai.vision.core import show_image, show_titled_image
from torchvision import transforms


class Denormalize:
    def __init__(self, mean, std):

        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]

        self.denormalize = transforms.Normalize(mean=new_mean, std=new_std)

    def __call__(self, tensor):
        return self.denormalize(tensor)


def plot_img(img, label=None, info=None, ax=None):

    def add_info_text(ax, info):
        ax.text(
            0.5, -0.1, info, transform=ax.transAxes, ha="center", va="top", fontsize=10
        )
        ax.xaxis.set_label_position("top")

    # using show_image from fastai to handle different image types
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(5, 5))

    if label:
        title = f"Label: {label}"
        show_titled_image((img, title), ax=ax)
    else:
        show_image(img, ax=ax)

    if info:
        # Add info as text below the image
        add_info_text(ax, info)

    if created_fig:
        plt.show()


def get_grid(num_rows, num_cols, figsize=(16, 8)):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows == 1:
        axes = [axes]  # Ensure axes is iterable
    elif num_cols == 1:
        axes = [[ax] for ax in axes]  # Ensure 2D list
    return fig, axes


def print_data_folder_structure(root_dir, max_depth=1):
    """Print the folder structure of the dataset directory."""
    config_tree = {
        "dirPath": root_dir,
        "onlyDirs": False,
        "maxDepth": max_depth,
        "sortBy": 1,  # Sort by type (files first, then folders)
    }
    DisplayTree(**config_tree)


def explore_extensions(root_dir):
    """Explore and print the file extensions in the dataset directory."""
    extensions = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in extensions:
                extensions[ext] = []
            extensions[ext].append(os.path.join(dirpath, filename))
    return extensions


def visual_exploration(dataset, num_rows=2, num_cols=4):
    """Visual exploration of the dataset by displaying random samples in a grid."""
    # Calculate total number of samples to display
    total_samples = num_rows * num_cols

    # Randomly select indices from the dataset without replacement
    # This ensures we get a diverse sample of the dataset
    indices = np.random.choice(len(dataset), total_samples, replace=False)

    # Create a grid of subplots with appropriate figure size
    # Each subplot gets (3 x 4) inches per image for good visibility
    fig, axes = get_grid(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 4))

    # Iterate through each axis and corresponding random index
    for ax, idx in zip(axes.flatten(), indices):
        # Load image and label from dataset at the random index
        image, label = dataset[idx]

        # Get human-readable description for the label
        description = dataset.get_label_description(label)

        # Create a combined label string with both number and description
        label = f"{label} - {description}"

        # Create info string showing index and image dimensions
        info = f"Index: {idx} Size: {image.size}"

        # Plot the image on the current axis with label and info
        plot_img(image, label=label, info=info, ax=ax)

    # Display the complete grid of images
    plt.show()

import copy
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader



def load_cifar100_subset(target_classes, train_transform, val_transform, root='./cifar_100'):
    """
    Loads and filters the CIFAR-100 dataset to include only specified target classes.

    This function first checks for a local copy of the CIFAR-100 dataset and
    downloads it if not found. It then filters both the training and test sets
    to retain only the images and labels corresponding to the classes specified
    in `target_classes`. The labels are remapped to be contiguous from 0.

    Args:
        target_classes: A list of class name strings to be included in the dataset subset.
        train_transform: A torchvision transform to be applied to the training dataset images.
        val_transform: A torchvision transform to be applied to the test/val dataset images.
        root: The root directory where the dataset is stored or will be downloaded.

    Returns:
        A tuple containing the filtered training dataset and the filtered test dataset.
        Returns (None, None) if a specified target class is not found.
    """
    # Construct the path to the CIFAR-100 dataset directory.
    cifar100_path = os.path.join(root, 'cifar-100-python')
    # Check if the dataset directory exists locally.
    dataset_found = os.path.isdir(cifar100_path)
    if dataset_found:
        print(f"Dataset found in '{root}'. Loading from local files.")
    # If not found, inform the user that it will be downloaded.
    else:
        print(f"Dataset not found in '{root}'. Downloading...")

    # Load the full CIFAR-100 training dataset.
    train_dataset_full = torchvision.datasets.CIFAR100(
        root=root,
        train=True,
        download=not dataset_found,
        transform=train_transform
    )

    # Load the full CIFAR-100 test dataset.
    test_dataset_full = torchvision.datasets.CIFAR100(
        root=root,
        train=False,
        download=not dataset_found,
        transform=val_transform
    )
    print("Dataset loaded successfully.")

    # Get the list of all class names from the dataset.
    all_classes = train_dataset_full.classes
    try:
        # Get the original integer indices for the target class names.
        target_indices = [all_classes.index(cls) for cls in target_classes]
    # Handle the case where a specified class name is not in the dataset.
    except ValueError as e:
        print(f"Error: One of the target classes not found in CIFAR-100. {e}")
        return None, None
        
    # Create a mapping from the original class indices to new, contiguous indices (0, 1, 2, ...).
    label_map = {old_label: new_label for new_label, old_label in enumerate(target_indices)}

    # Define a helper function to filter a dataset based on the target classes.
    def _filter_dataset(dataset):
        # Convert the list of targets to a NumPy array for efficient boolean indexing.
        targets_np = np.array(dataset.targets)
        # Create a boolean mask to identify which samples belong to the target classes.
        indices_to_keep = np.isin(targets_np, target_indices)
        
        # Filter the dataset's image data using the boolean mask.
        dataset.data = dataset.data[indices_to_keep]
        
        # Get the original labels of the samples that are being kept.
        original_targets_to_keep = targets_np[indices_to_keep]
        # Remap the original labels to the new contiguous labels.
        dataset.targets = [label_map[target] for target in original_targets_to_keep]
        
        # Update the dataset's class list to only include the target classes.
        dataset.classes = target_classes
        return dataset

    print(f"\nFiltering for {len(target_classes)} classes...")
    # Apply the filtering logic to the full training dataset.
    train_dataset_subset = _filter_dataset(train_dataset_full)
    # Apply the filtering logic to the full test dataset.
    test_dataset_subset = _filter_dataset(test_dataset_full)
    print("Filtering complete. Returning training and validation datasets.")
    
    # Return the filtered training and test subsets.
    return train_dataset_subset, test_dataset_subset



def visualise_images(dataset, grid):
    """
    Displays a grid of images from a dataset, with one random image per class.

    Args:
        dataset: The dataset object containing the images and labels.
        grid (tuple): A tuple specifying the number of rows and columns for the image grid.
    """

    # Create a shallow copy of the dataset to avoid modifying the original
    dataset_copy = copy.copy(dataset)
    # Set the transform on the copied dataset to convert images to tensors
    dataset_copy.transform = torchvision.transforms.ToTensor()

    # Create a DataLoader to handle batching and shuffling of the data
    loader = DataLoader(dataset_copy, batch_size=64, shuffle=True)

    # Unpack the grid dimensions from the input tuple
    rows, cols = grid
    # Calculate the total number of images to display in the grid
    num_images_to_show = rows * cols

    # Get the dataset object from the DataLoader
    dataset_to_show = loader.dataset

    # Create a dictionary to store lists of indices for each class
    class_indices = defaultdict(list)
    # Iterate through the dataset to populate the class_indices dictionary
    for idx, target in enumerate(dataset_to_show.targets):
        class_indices[target].append(idx)
        
    # Get the list of class names from the dataset
    class_names = dataset_to_show.classes

    # Create a figure and a set of subplots for the grid layout
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Iterate over each subplot in the grid
    for i, ax in enumerate(axes.flat):
        # If the current index is out of bounds, turn off the subplot axis
        if i >= num_images_to_show or i >= len(class_names):
            ax.axis('off')
            continue
            
        # Set the class label based on the current iteration index
        class_label = i
        
        # Get the list of image indices for the current class
        indices_for_class = class_indices[class_label]
        # If there are no images for this class, turn off the subplot axis
        if not indices_for_class:
            ax.axis('off')
            continue

        # Choose a random image index from the list for the current class
        random_image_index = random.choice(indices_for_class)
        
        # Retrieve the image tensor and its corresponding label from the dataset
        image_tensor, _ = dataset_to_show[random_image_index]
        
        # Convert the tensor to a NumPy array and transpose dimensions for display
        img_to_display = image_tensor.numpy().transpose((1, 2, 0))
        
        # Get the name of the class corresponding to the class label
        class_name = class_names[class_label]
        
        # Display the image on the current subplot
        ax.imshow(img_to_display)
        
        # Set the title of the subplot to the capitalized class name
        ax.set_title(class_name.capitalize(), fontsize=16)
        # Turn off the axis for a cleaner look
        ax.axis('off')

    # Adjust subplot parameters for a tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Display the plot
    plt.show()

    # Clean up the copied dataset to free up memory
    del dataset_copy
    
    

def plot_training_metrics(metrics):
    """
    Plots the training and validation metrics from a model training process.

    This function generates two side-by-side plots:
    1. Training Loss vs. Validation Loss.
    2. Validation Accuracy.

    Args:
        metrics (list): A list or tuple containing three lists:
                        [train_losses, val_losses, val_accuracies].
    """
    # Unpack the metrics into their respective lists
    train_losses, val_losses, val_accuracies = metrics
    
    # Determine the number of epochs from the length of the training losses list
    num_epochs = len(train_losses)
    # Create a 1-indexed range of epoch numbers for the x-axis
    epochs = range(1, num_epochs + 1)

    # Create a figure and a set of subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Configure the first subplot for training and validation loss ---
    # Select the first subplot
    ax1 = axes[0]
    # Plot training loss data
    ax1.plot(epochs, train_losses, color='#085c75', linewidth=2.5, marker='o', markersize=5, label='Training Loss')
    # Plot validation loss data
    ax1.plot(epochs, val_losses, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Loss')
    # Set the title and axis labels for the loss plot
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    # Display the legend
    ax1.legend()
    # Add a grid for better readability
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Configure the second subplot for validation accuracy ---
    # Select the second subplot
    ax2 = axes[1]
    # Plot validation accuracy data
    ax2.plot(epochs, val_accuracies, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Accuracy')
    # Set the title and axis labels for the accuracy plot
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    # Display the legend
    ax2.legend()
    # Add a grid for better readability
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # --- Apply dynamic and consistent styling to both subplots ---
    # Calculate a suitable interval for the x-axis ticks to avoid clutter
    x_interval = (num_epochs - 1) // 10 + 1

    # Loop through each subplot to apply common axis settings
    for ax in axes:
        # Set the y-axis to start at 0 and the x-axis to span the epochs
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=1, right=num_epochs)
        
        # Set the major tick locator for the x-axis using the dynamic interval
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_interval))
        # Set the font size for the tick labels on both axes
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Adjust subplot parameters for a tight layout
    plt.tight_layout()
    # Display the plots
    plt.show()
    
    
    
def visualise_predictions(model, data_loader, device, grid):
    """
    Visualizes model predictions on a grid of images from a dataset.

    Args:
        model: The trained PyTorch model to use for predictions.
        data_loader: The PyTorch DataLoader for the dataset.
        device: The device (e.g., 'cpu' or 'cuda') to run the model on.
        grid (tuple): A tuple specifying the number of rows and columns for the image grid.
    """
    # Set the model to evaluation mode
    model.eval()

    # Get the dataset and class names from the data loader
    dataset = data_loader.dataset
    class_names = dataset.classes
    
    # Define mean and standard deviation values for de-normalizing the images
    cifar100_mean = np.array([0.5071, 0.4867, 0.4408])
    cifar100_std = np.array([0.2675, 0.2565, 0.2761])
    
    # Create a dictionary to store lists of indices for each class
    class_indices = defaultdict(list)
    # Iterate through the dataset to populate the class_indices dictionary
    for idx, target in enumerate(dataset.targets):
        class_indices[target].append(idx)
        
    # Unpack the grid dimensions
    rows, cols = grid
    # Calculate the total number of images to display
    num_images_to_show = rows * cols
    
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2)) 
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.8)

    # Iterate over each subplot in the grid
    for i, ax in enumerate(axes.flat):
        # If the current index is out of bounds, turn off the subplot axis
        if i >= num_images_to_show or i >= len(class_names):
            ax.axis('off')
            continue
            
        # Set the class label based on the current iteration index
        class_label = i
        
        # Get the list of image indices for the current class
        indices_for_class = class_indices[class_label]
        # If there are no images for this class, turn off the subplot axis
        if not indices_for_class:
            ax.axis('off')
            continue

        # Choose a random image index from the list for the current class
        random_image_index = random.choice(indices_for_class)
        # Retrieve the image tensor and its true label
        image_tensor, true_label = dataset[random_image_index]
        
        # Add a batch dimension and move the tensor to the specified device
        image_batch = image_tensor.unsqueeze(0).to(device)
        
        # Disable gradient calculations for inference
        with torch.no_grad():
            # Get model predictions
            output = model(image_batch)
            # Find the index of the highest score, which is the predicted class
            _, predicted_index = torch.max(output, 1)
        
        # Extract the predicted label as a Python number
        predicted_label = predicted_index.item()
        
        # Convert tensor to a NumPy array and transpose dimensions for display
        img_np = image_tensor.cpu().numpy().transpose((1, 2, 0))
        # De-normalize the image using the predefined mean and std
        denormalized_img = cifar100_std * img_np + cifar100_mean
        # Clip the pixel values to the valid range [0, 1]
        clipped_img = np.clip(denormalized_img, 0, 1)
        
        # Get the string names for the true and predicted labels
        true_name = class_names[true_label]
        predicted_name = class_names[predicted_label]
        
        # Set the title color to green for correct predictions and red for incorrect ones
        title_color = 'green' if true_label == predicted_label else 'red'
        
        # Display the image
        ax.imshow(clipped_img)
        # Set the title with true and predicted labels
        ax.set_title(f"True: {true_name.capitalize()}\nPred: {predicted_name.capitalize()}", 
                     color=title_color, fontsize=10, pad=5)
        # Turn off the axis
        ax.axis('off')

    # Adjust subplot parameters for a tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Show the final plot
    plt.show()
    
    
import itertools
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset



def load_cifar100_subset(target_classes, train_transform, val_transform, root='./cifar_100'):
    """
    Loads and filters the CIFAR-100 dataset to include only specified target classes.

    This function first checks for a local copy of the CIFAR-100 dataset and
    downloads it if not found. It then filters both the training and test sets
    to retain only the images and labels corresponding to the classes specified
    in `target_classes`. The labels are remapped to be contiguous from 0.

    Args:
        target_classes: A list of class name strings to be included in the dataset subset.
        train_transform: A torchvision transform to be applied to the training dataset images.
        val_transform: A torchvision transform to be applied to the test/val dataset images.
        root: The root directory where the dataset is stored or will be downloaded.

    Returns:
        A tuple containing the filtered training dataset and the filtered test dataset.
        Returns (None, None) if a specified target class is not found.
    """
    # Construct the path to the CIFAR-100 dataset directory.
    cifar100_path = os.path.join(root, 'cifar-100-python')
    # Check if the dataset directory exists locally.
    if os.path.isdir(cifar100_path):
        print(f"Dataset found in '{root}'. Loading from local files.")
    # If not found, inform the user that it will be downloaded.
    else:
        print(f"Dataset not found in '{root}'. Downloading...")

    # Load the full CIFAR-100 training dataset.
    train_dataset_full = torchvision.datasets.CIFAR100(
        root=root, 
        train=True, 
        download=True, 
        transform=train_transform
    )

    # Load the full CIFAR-100 test dataset.
    test_dataset_full = torchvision.datasets.CIFAR100(
        root=root, 
        train=False, 
        download=True, 
        transform=val_transform
    )
    print("Dataset loaded successfully.")

    # Get the list of all class names from the dataset.
    all_classes = train_dataset_full.classes
    try:
        # Get the original integer indices for the target class names.
        target_indices = [all_classes.index(cls) for cls in target_classes]
    # Handle the case where a specified class name is not in the dataset.
    except ValueError as e:
        print(f"Error: One of the target classes not found in CIFAR-100. {e}")
        return None, None
        
    # Create a mapping from the original class indices to new, contiguous indices (0, 1, 2, ...).
    label_map = {old_label: new_label for new_label, old_label in enumerate(target_indices)}

    # Define a helper function to filter a dataset based on the target classes.
    def _filter_dataset(dataset):
        # Convert the list of targets to a NumPy array for efficient boolean indexing.
        targets_np = np.array(dataset.targets)
        # Create a boolean mask to identify which samples belong to the target classes.
        indices_to_keep = np.isin(targets_np, target_indices)
        
        # Filter the dataset's image data using the boolean mask.
        dataset.data = dataset.data[indices_to_keep]
        
        # Get the original labels of the samples that are being kept.
        original_targets_to_keep = targets_np[indices_to_keep]
        # Remap the original labels to the new contiguous labels.
        dataset.targets = [label_map[target] for target in original_targets_to_keep]
        
        # Update the dataset's class list to only include the target classes.
        dataset.classes = target_classes
        return dataset

    print(f"\nFiltering for {len(target_classes)} classes...")
    # Apply the filtering logic to the full training dataset.
    train_dataset_subset = _filter_dataset(train_dataset_full)
    # Apply the filtering logic to the full test dataset.
    test_dataset_subset = _filter_dataset(test_dataset_full)
    print("Filtering complete. Returning training and validation datasets.")
    
    # Return the filtered training and test subsets.
    return train_dataset_subset, test_dataset_subset



def visualise_images(loader, grid):
    """
    Visualizes a grid of random images from a dataset, showing one image per class.

    Args:
        loader: The DataLoader object containing the dataset.
        grid: A tuple specifying the grid dimensions as (rows, cols).
    """
    # Unpack the grid dimensions from the input tuple
    rows, cols = grid
    # Calculate the total number of images to display in the grid
    num_images_to_show = rows * cols

    # Get the dataset object from the DataLoader
    dataset_to_show = loader.dataset

    # Create a dictionary to store lists of indices for each class
    class_indices = defaultdict(list)
    # Iterate through the dataset to populate the class_indices dictionary
    for idx, target in enumerate(dataset_to_show.targets):
        class_indices[target].append(idx)
        
    # Get the list of class names from the dataset
    class_names = dataset_to_show.classes

    # Create a figure and a set of subplots for the grid layout
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Iterate over each subplot in the grid
    for i, ax in enumerate(axes.flat):
        # If the current index is out of bounds, turn off the subplot axis
        if i >= num_images_to_show or i >= len(class_names):
            ax.axis('off')
            continue
            
        # Set the class label based on the current iteration index
        class_label = i
        
        # Get the list of image indices for the current class
        indices_for_class = class_indices[class_label]
        # If there are no images for this class, turn off the subplot axis
        if not indices_for_class:
            ax.axis('off')
            continue

        # Choose a random image index from the list for the current class
        random_image_index = random.choice(indices_for_class)
        
        # Retrieve the image tensor and its corresponding label from the dataset
        image_tensor, _ = dataset_to_show[random_image_index]
        
        # Convert the tensor to a NumPy array and transpose dimensions for display
        img_to_display = image_tensor.numpy().transpose((1, 2, 0))
        
        # Rescale the pixel values from the normalized range to [0, 1] for proper visualization
        min_val = img_to_display.min()
        max_val = img_to_display.max()
        img_to_display = (img_to_display - min_val) / (max_val - min_val)
        
        # Get the name of the class corresponding to the class label
        class_name = class_names[class_label]
        
        # Display the image on the current subplot
        ax.imshow(img_to_display)
        
        # Set the title of the subplot to the capitalized class name
        ax.set_title(class_name.capitalize(), fontsize=16)
        # Turn off the axis for a cleaner look
        ax.axis('off')

    # Adjust subplot parameters for a tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Display the plot
    plt.show()
    
    

def plot_training_metrics(metrics):
    """
    Plots the training and validation metrics from a model training process.

    This function generates two side-by-side plots:
    1. Training Loss vs. Validation Loss.
    2. Validation Accuracy.

    Args:
        metrics (list): A list or tuple containing three lists:
                        [train_losses, val_losses, val_accuracies].
    """
    # Unpack the metrics into their respective lists
    train_losses, val_losses, val_accuracies = metrics
    
    # Determine the number of epochs from the length of the training losses list
    num_epochs = len(train_losses)
    # Create a 1-indexed range of epoch numbers for the x-axis
    epochs = range(1, num_epochs + 1)

    # Create a figure and a set of subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Configure the first subplot for training and validation loss ---
    # Select the first subplot
    ax1 = axes[0]
    # Plot training loss data
    ax1.plot(epochs, train_losses, color='#085c75', linewidth=2.5, marker='o', markersize=5, label='Training Loss')
    # Plot validation loss data
    ax1.plot(epochs, val_losses, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Loss')
    # Set the title and axis labels for the loss plot
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    # Display the legend
    ax1.legend()
    # Add a grid for better readability
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Configure the second subplot for validation accuracy ---
    # Select the second subplot
    ax2 = axes[1]
    # Plot validation accuracy data
    ax2.plot(epochs, val_accuracies, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Accuracy')
    # Set the title and axis labels for the accuracy plot
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    # Display the legend
    ax2.legend()
    # Add a grid for better readability
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # --- Apply dynamic and consistent styling to both subplots ---
    # Calculate a suitable interval for the x-axis ticks to avoid clutter
    x_interval = (num_epochs - 1) // 10 + 1

    # Loop through each subplot to apply common axis settings
    for ax in axes:
        # Set the y-axis to start at 0 and the x-axis to span the epochs
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=1, right=num_epochs)
        
        # Set the major tick locator for the x-axis using the dynamic interval
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_interval))
        # Set the font size for the tick labels on both axes
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Adjust subplot parameters for a tight layout
    plt.tight_layout()
    # Display the plots
    plt.show()
    
    
    
def verify_training_process(model_class, train_loader, loss_function, train_epoch_fn, device):
    """
    Verifies the training process on a small subset of data for a few epochs.

   Args:
        model_class: The model class to be instantiated for verification.
        train_loader: The DataLoader for the training dataset.
        loss_function: The loss function to be used during training.
        train_epoch_fn: The function that executes one training epoch.
        device: The device (e.g., 'cuda' or 'cpu') to run the verification on.
    """
    # Print a header for the verification process
    print("--- Verifying train_epoch (training for 5 epochs) ---\n")

    # Define the number of epochs and batches for the verification run
    NUM_VERIFY_EPOCHS = 5
    NUM_VERIFY_BATCHES = 10

    # Instantiate the model and move it to the specified device
    verify_model = model_class(15).to(device)
    # Initialize the Adam optimizer with a learning rate
    verify_optimizer = optim.Adam(verify_model.parameters(), lr=0.0005)

    # Create a small subset of the training data for quick verification
    batches = list(itertools.islice(iter(train_loader), NUM_VERIFY_BATCHES))
    # Concatenate the images and labels from the selected batches
    all_images = torch.cat([b[0] for b in batches])
    all_labels = torch.cat([b[1] for b in batches])
    # Create a TensorDataset and a DataLoader for the subset
    verify_subset_dataset = TensorDataset(all_images, all_labels)
    verify_subset_loader = DataLoader(verify_subset_dataset, batch_size=train_loader.batch_size)

    # Clone the initial weights of a specific layer to check for changes later
    initial_weight = verify_model.conv_block1.block[0].weight.clone()
    # Initialize a list to store the loss from each epoch
    epoch_losses = []

    print(f"Training on {len(verify_subset_dataset)} images for {NUM_VERIFY_EPOCHS} epochs:\n")
    # Loop through the defined number of epochs for verification
    for epoch in range(NUM_VERIFY_EPOCHS):
        # Run a single training epoch and get the loss
        loss = train_epoch_fn(
            model=verify_model,
            train_loader=verify_subset_loader,
            loss_function=loss_function,
            optimizer=verify_optimizer,
            device=device
        )
        # Append the loss to the list and print the epoch's result
        epoch_losses.append(loss)
        print(f"Epoch [{epoch+1}/{NUM_VERIFY_EPOCHS}], Loss: {loss:.4f}")

    # Get the weights of the same layer after training has completed
    trained_weight = verify_model.conv_block1.block[0].weight

    # Check if the weights have changed from their initial values
    weights_changed = not torch.equal(initial_weight, trained_weight)
    if weights_changed:
        print("\nWeight Update Check:\tModel weights changed during training.")
    else:
        print("\nWeight Update Check:\tModel weights DID NOT change.")

    # Check if the final loss is less than the initial loss
    loss_decreased = epoch_losses[-1] < epoch_losses[0]
    if loss_decreased:
        print(f"Loss Trend Check:\tLoss decreased from {epoch_losses[0]:.4f} to {epoch_losses[-1]:.4f}.")
    else:
        print(f"Loss Trend Check:\tLoss DID NOT show a decreasing trend.")
        
        
        
def verify_validation_process(model_class, val_loader, loss_function, validate_epoch_fn, device):
    """
    Verifies the validation process on a small subset of data.

    Args:
        model_class: The model class to be instantiated for verification.
        val_loader: The DataLoader for the validation dataset.
        loss_function: The loss function to be used during validation.
        validate_epoch_fn: The function that executes one validation epoch.
        device: The device (e.g., 'cuda' or 'cpu') to run the verification on.
    """
    # Print a header for the verification process
    print("--- Verifying validate_epoch ---\n")

    # Define the number of batches for the verification run
    NUM_VERIFY_BATCHES = 10

    # Instantiate the model and move it to the specified device
    verify_model = model_class(15).to(device)

    # Create a small subset of the validation data for quick verification
    val_batches = list(itertools.islice(iter(val_loader), NUM_VERIFY_BATCHES))
    # Concatenate the images and labels from the selected batches
    val_all_images = torch.cat([b[0] for b in val_batches])
    val_all_labels = torch.cat([b[1] for b in val_batches])
    # Create a TensorDataset and a DataLoader for the subset
    verify_val_subset_dataset = TensorDataset(val_all_images, val_all_labels)
    verify_val_subset_loader = DataLoader(verify_val_subset_dataset, batch_size=val_loader.batch_size)

    # Clone the initial weights of a specific layer to check for changes
    initial_weight = verify_model.conv_block1.block[0].weight.clone()

    print(f"Validating on {len(verify_val_subset_dataset)} images:\n")
    # Run a single validation epoch on the subset and get the outputs
    val_loss, val_accuracy = validate_epoch_fn(
        model=verify_model,
        val_loader=verify_val_subset_loader,
        loss_function=loss_function,
        device=device
    )

    # Get the weights of the same layer after the validation function has run
    validated_weight = verify_model.conv_block1.block[0].weight

    # Print the returned loss and accuracy
    print(f"Returned Validation Loss: {val_loss:.4f}")
    print(f"Returned Validation Accuracy: {val_accuracy:.2f}%\n")

    # Check if the returned loss and accuracy are of the correct data type (float)
    types_correct = isinstance(val_loss, float) and isinstance(val_accuracy, float)
    if types_correct:
        print("\nReturn Types Check:\tFunction returned a float for loss and accuracy.")
    else:
        print("\nReturn Types Check:\tFunction DID NOT return the correct data types.")

    # Check that the weights have not changed during validation
    weights_unchanged = torch.equal(initial_weight, validated_weight)
    if weights_unchanged:
        print("Weight Integrity Check:\tModel weights were not changed during validation.")
    else:
        print("Weight Integrity Check:\tModel weights WERE CHANGED.")