import cv2 as cv # will be used for a number of image processing methods
import os # used for file path checks
import numpy as np

def Img_Conv(img_source: str):
    """
    This will produce a binary image and then save said image

    img_source: our image file's path
    """ 

    # Supervior: you may want to compare images with those produced from Fiji as they may produce results that can be compared
    

    # read the image file
    img = cv.imread(img_source, 2)

    #unsure why the error is present as code works as expected may ignore if not issues arise
    ret, bw_img = cv.threshold(img, 127, 255, cv.THRESH_TOZERO) #cv.THRESH_BINARY or  to make it binary

    cv.imshow("Binary", bw_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return bw_img

def Calc_img_Moment(img, p: int, q: int):
    """
    Calculates the image moment for our picture of interest
    
    img: our image of interest

    p: the value the x coordinates are put to the power to

    q: the value the y coordinates are put to the power to

    returns the image moment of interest
    """

    order = p + q

    
    if p < 0 or q < 0:
        return "please give me a proper value input"
    
    elif order <= 1:
        h, w = img.shape
        y, x = np.mgrid[0:h, 0:w]
        return np.sum((x**p) * (y**q) * img)

    else:
        h, w = img.shape
        y, x = np.mgrid[0:h, 0:w]
        x_cent, y_cent = Centroid(img)
        return np.sum( ((x - x_cent)**p) * ((y - y_cent)**q) * img)
    
def Centroid(img):
    """
    The calculation of the centroid is done here. returns the coordinates of the centroid

    img: the image of interest

    returns the coordinates for the centroid
    """

    M_00 = Calc_img_Moment(img,0,0)
    M_10 = Calc_img_Moment(img,1,0)
    M_01 = Calc_img_Moment(img,0,1)

    return M_10/M_00, M_01/M_00