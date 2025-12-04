import Img_moments as IM
import os
import cv2 as cv # will be used for a number of image processing methods
import numpy as np

# Here the code used is that which will help make the image a "good" binary image
# good being defined as: lacking any dark spots or a minimal number of them while only containing white or black pixels


def Img_Conv(img_source: str, inv: bool):
    """
    This will produce a binary image and then save said image

    img_source: our image file's path
    """

    # Supervior: you may want to compare images with those produced from Fiji as they may produce results that can be compared

    # os.chdir("") maybe change the directory??

    # read the image file
    img = cv.imread(img_source, 2)

    #window_size = 100
    #fine_tune = 2
    color = 255

    # unsure why the error is present as code works as expected may ignore if not issues arise
    if inv == True:
        ret, bw_img = cv.threshold(img, 127, color, cv.THRESH_BINARY_INV) # cv.THRESH_BINARY or  to make it binary

        # bw_img = cv.adaptiveThreshold(img, color, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, window_size, fine_tune) this is a method which doesn't work for filling gaps
    else:
        ret, bw_img = cv.threshold(img, 127, color, cv.THRESH_BINARY) # cv.THRESH_BINARY or  to make it binary
        #bw_img = cv.adaptiveThreshold(img, color, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, window_size, fine_tune)

    cv.imshow("Binary", bw_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return bw_img

def center_image_to_max_distance(img):
    """
    Centres an image so that the distance from the centroid to each border
    equals the maximum original distance to any border.

    Parameters:
        img (np.ndarray): Grayscale or color image (H x W or H x W x C)

    Returns:
        centered_img (np.ndarray): Padded image with the centroid centered
    """
    h, w = img.shape[:2]

    # Compute image moments
    M = cv.moments(img if len(img.shape) == 2 else cv.cvtColor(img, cv.COLOR_BGR2GRAY))

    # Centroid (cx, cy)
    if M["m00"] == 0:
        raise ValueError("Image has zero mass (all zeros); cannot compute centroid.")
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # Distances from centroid to each border
    left = cx
    right = w - cx
    top = cy
    bottom = h - cy

    # Find max distance
    dmax = int(np.ceil(max(left, right, top, bottom)))

    # Compute required padding on each side to make all distances = dmax
    pad_left = int(np.ceil(dmax - left))
    pad_right = int(np.ceil(dmax - right))
    pad_top = int(np.ceil(dmax - top))
    pad_bottom = int(np.ceil(dmax - bottom))

    # Pad image symmetrically
    centered_img = cv.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType = cv.BORDER_CONSTANT,
        value=0,  # black background
    )

    return centered_img

def recenter_binary_image(binary_img, centroid):
    """
    Recenter a binary image so that the given centroid becomes the center,
    padding as needed to keep the entire image visible.

    Parameters
    ----------
    binary_img : np.ndarray
        Binary image (single-channel, 0/255 or 0/1).
    centroid : tuple(int, int)
        (cx, cy) coordinates of the centroid.

    Returns
    -------
    recentered : np.ndarray
        Binary image with the centroid at the center.
    """
    # Ensure single-channel binary
    if len(binary_img.shape) > 2:
        raise ValueError("Input image must be single-channel (binary).")

    h, w = binary_img.shape
    cx, cy = centroid

    # Distances from centroid to edges
    left = cx
    right = w - cx
    top = cy
    bottom = h - cy

    # Maximum distance from centroid to any border
    max_dist = int(max(left, right, top, bottom))

    # New (square) image size
    new_size = 2 * max_dist

    # Center of new image
    new_cx, new_cy = new_size // 2, new_size // 2

    # Compute offset for placing original image
    paste_x = new_cx - cx
    paste_y = new_cy - cy

    # Create new padded binary image (background = 0)
    recentered = np.zeros((new_size, new_size), dtype=binary_img.dtype)

    # Paste the original binary image into the padded frame
    recentered[paste_y:paste_y + h, paste_x:paste_x + w] = binary_img

    return recentered

def fill_black_spots(binary, file_path, do_not):
    if file_path not in do_not:
        kernel = np.ones((3,3), np.uint8)
        binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=2)
        filled = cv.dilate(binary, kernel, iterations=1)
    else:
        filled = binary
    return filled