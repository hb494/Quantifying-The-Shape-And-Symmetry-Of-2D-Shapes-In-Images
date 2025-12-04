import matplotlib.pyplot as plt
import Helpers as H
import os
import cv2 as cv # will be used for a number of image processing methods
import numpy as np
import math

def draw_all_symmetry_lines(original_image, centroid, angles, colour=(0, 255, 0), thickness=2):
    """
    Draws all symmetry lines (given by angles in degrees) through the centroid
    on a clean copy of the image (no black streak artifacts).
    """

    cx, cy = centroid
    height, width = original_image.shape[:2]
    length = max(height, width) * 2

    # Work on a clean colour version of the image
    if len(original_image.shape) == 2:
        img_with_lines = cv.cvtColor(original_image, cv.COLOR_GRAY2BGR)
    else:
        img_with_lines = original_image.copy()

    # Draw centroid (red)
    #cv.circle(img_with_lines, (cx, cy), 4, (0, 0, 255), -1)

    # Draw each symmetry line separately on the same clean copy
    for theta_deg in angles:
        theta = math.radians(theta_deg)
        dx = math.sin(theta)
        dy = math.cos(theta)

        x1 = int(cx - length * dx)
        y1 = int(cy - length * dy)
        x2 = int(cx + length * dx)
        y2 = int(cy + length * dy)

        # Draw the line in green (or chosen colour)
        cv.line(img_with_lines, (x1, y1), (x2, y2), colour, thickness)

    return img_with_lines

def plot_degrees_vs_scores(degrees, scores, filename, xlabel="Angle from vertical axis (Degrees)", ylabel="Scores", Hu=False, save_dir=r"C:\Users\alloh\OneDrive\Desktop\coding\workspace\Term One Labs\Data\Plots"):
    """
    Creates and saves a graph of degrees against scores.

    Parameters
    ----------
    degrees : list or array-like
        X-axis data representing degrees.
    scores : list or array-like
        Y-axis data representing scores.
    title : str, optional
        Title of the graph. Default is "Degrees vs Scores".
    xlabel : str, optional
        Label for the x-axis. Default is "Degrees".
    ylabel : str, optional
        Label for the y-axis. Default is "Scores".
    filename : str, optional
        Name of the file to save. Default is "degrees_vs_scores.png".
    save_dir : str, optional
        Directory where the file should be saved. Default is the current directory.

    Returns
    -------
    None
    """
    # Check that both lists have the same length
    if len(degrees) != len(scores):
        raise ValueError("The length of 'degrees' and 'scores' must be the same.")
    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    if not Hu:
        title = f"Degrees vs Symmetry Score for {filename} without Hu moments"
    else:
        title = f"Degrees vs Symmetry Score for {filename} with Hu moments"
    
    new_name = ""
    i = 0
    # create the new file name
    while filename[i] != ".":
        new_name += filename[i]
        i += 1
    if Hu:
        new_name += f"_Hu_Plot{filename[i:]}"
    else:
        new_name += f"_Plot{filename[i:]}"

    
    # Construct full file path
    filepath = os.path.join(save_dir, new_name)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(degrees, scores, marker='o', linestyle='-', linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to the specified location
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Plot saved as '{filepath}'")