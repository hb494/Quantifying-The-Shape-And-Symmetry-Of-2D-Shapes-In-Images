import numpy as np

def Raw_Img_Moment(img, p: int, q: int):
    """
    Calculates the raw image moment for our picture of interest
    
    img: our image of interest

    p: the value the x coordinates are put to the power to

    q: the value the y coordinates are put to the power to

    returns the image moment of interest or zero depending on inputs
    """

    order = p + q

    
    if order >= 2:
        print("please give me a proper value input, positive values")
        return 0
    
    elif order <= 1:
        h, w = img.shape
        y, x = np.mgrid[0:h, 0:w]
        return np.sum((x**p) * (y**q) * img)
    
    else:
        print("please give me integer orders less than 2")
        return 0

def Central_Order_Moments(img, p: int, q: int, x_cent, y_cent, normal = False):

    """
    This calculates the central image moments.

    img: our image of interest

    p: the value the x coordinates are put to the power to

    q: the value the y coordinates are put to the power to

    x_cent: x coordinate of the centroid

    y_cent: y coordinate of the centroid

    normal: chooses if the code will run with or without normalising the moments

    returns the image moment of interest either normalised or not.

    """

    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w]

    if normal == False: 
        # standard version
        return np.sum( ((x - x_cent)**p) * ((y - y_cent)**q) * img)
    
    else:
        #normalised version of the moments
        return (np.sum( ((x - x_cent)**p) * ((y - y_cent)**q) * img))/((np.sum(((x - x_cent)**0) * ((y - y_cent)**0) * img)))**(1 + (p+q)/2)


def Centroid(img):
    """
    The calculation of the centroid is done here. returns the coordinates of the centroid

    img: the image of interest

    returns the coordinates for the centroid
    """
    
    M_00 = Raw_Img_Moment(img,0,0)
    M_10 = Raw_Img_Moment(img,1,0)
    M_01 = Raw_Img_Moment(img,0,1)

    x_mean = M_10/M_00
    y_mean = M_01/M_00

    return x_mean, y_mean


def Hu_Moments(img, x_cent, y_cent):
    
    # Normalised central moments
    n20 = Central_Order_Moments(img, 2, 0, x_cent, y_cent, normal=True)
    n02 = Central_Order_Moments(img, 0, 2, x_cent, y_cent, normal=True)
    n11 = Central_Order_Moments(img, 1, 1, x_cent, y_cent, normal=True)
    n30 = Central_Order_Moments(img, 3, 0, x_cent, y_cent, normal=True)
    n12 = Central_Order_Moments(img, 1, 2, x_cent, y_cent, normal=True)
    n21 = Central_Order_Moments(img, 2, 1, x_cent, y_cent, normal=True)
    n03 = Central_Order_Moments(img, 0, 3, x_cent, y_cent, normal=True)

    # Hu's seven invariant moments
    hu1 = n20 + n02
    hu2 = (n20 - n02)**2 + 4*(n11**2)
    hu3 = (n30 - 3*n12)**2 + (3*n21 - n03)**2
    hu4 = (n30 + n12)**2 + (n21 + n03)**2
    hu5 = ((n30 - 3*n12)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2)
          + (3*n21 - n03)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2))
    hu6 = ((n20 - n02)*((n30 + n12)**2 - (n21 + n03)**2)
          + 4*n11*(n30 + n12)*(n21 + n03))
    hu7 = ((3*n21 - n03)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2)
          - (n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2))

    return np.array([hu1, hu2, hu3, hu4, hu5, hu6, hu7])