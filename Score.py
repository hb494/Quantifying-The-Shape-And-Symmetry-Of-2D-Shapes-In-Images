import Img_moments as IM
import cv2 as cv # will be used for a number of image processing methods
import numpy as np
import math

def orientation(Mu_11, Mu_20, Mu_02):
    const = 1e-15
    return 0.5 * np.arctan((2*Mu_11)/(Mu_20 - Mu_02 + const))

def eccentricity(Mu_02, Mu_20, Mu_00):

    C_00 = Mu_20/Mu_00
    C_11 = Mu_02/Mu_00

    c = C_11*C_00
    a = 1
    b = - (C_11 + C_00)

    # calculate the discriminant
    d = (b**2) - (4*a*c)

    # find two solutions
    lambda_1 = (-b-np.sqrt(d))/(2*a)
    lambda_2 = (-b+np.sqrt(d))/(2*a)

    if lambda_1 > lambda_2:
        return np.sqrt(1 - (lambda_2/lambda_1))
    else:
        return np.sqrt(1 - (lambda_1/lambda_2))
    

def Spread(Mu_20, Mu_02, Mu_00):
    const = 1e-15
    return (Mu_20 + Mu_02)/(Mu_00 + const)

def Skewness(Mu_30, Mu_03, Mu_20, Mu_02):
    return (Mu_30/((Mu_20)**(1.5))), (Mu_03/((Mu_02)**1.5))

def find_local_minima(values):
    """
    Finds all local minima in a sequence of numeric values.
    A local minimum is a value that is lower than the value before
    and the value after it.

    Parameters:
        values (list or array): Sequence of numeric values.

    Returns:
        list: Indices where local minima occur.
    """

    minima_indices = []

    # Cannot have minima at the first or last position
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            minima_indices.append(i)

    return minima_indices


def find_symmetry_binary(binary_image, centroid, height, width, Hu=False, looping = False):

    #symm_cut_off = 0.99, is used for filtering

    lines_symm = []
    scores = []
    moments_of_interest = []

    cx, cy = centroid

    for theta_deg in range(0, 181):
        # Rotate the binary image around the centroid
        rotation_matrix = cv.getRotationMatrix2D((cx, cy), -theta_deg, 1.0)
        rotated_binary = cv.warpAffine(binary_image.copy(), rotation_matrix, (width, height), flags=cv.INTER_NEAREST,borderValue=0)

        # to remove artifacts
        _, rotated_binary = cv.threshold(rotated_binary, 127, 255, cv.THRESH_BINARY)
        
        # DEBUG: Show the rotated image
        #cv.imshow(f"Rotated {theta_deg} degrees", rotated_binary)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        
        # The symmetry axis is now VERTICAL through the CENTER of the rotated image
        rot_height, rot_width = rotated_binary.shape
        #only works due to the recentering used.
        center_x = rot_width // 2
        
        # Split into left and right halves around the vertical center
        left_half = rotated_binary[:, :center_x]
        right_half = rotated_binary[:, center_x:]
        
        # Flip the right half for comparison
        right_half_flipped = cv.flip(right_half, 1)  # Flip horizontally
        
        # Ensure both halves have the same dimensions
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_cropped = left_half[:, -min_width:] if left_half.shape[1] > min_width else left_half
        right_cropped = right_half_flipped[:, :min_width] if right_half_flipped.shape[1] > min_width else right_half_flipped
        
        if not Hu:
            # Calculate moments for comparison
            M_left = {
                "m00": IM.Raw_Img_Moment(left_cropped, 0, 0),
                "m10": IM.Raw_Img_Moment(left_cropped, 1, 0),
                "m01": IM.Raw_Img_Moment(left_cropped, 0, 1)
            }
            M_right = {
                "m00": IM.Raw_Img_Moment(right_cropped, 0, 0),
                "m10": IM.Raw_Img_Moment(right_cropped, 1, 0),
                "m01": IM.Raw_Img_Moment(right_cropped, 0, 1)
            }
            
            # Calculate symmetry score based on area similarity
            total_area = M_left["m00"] + M_right["m00"]

            if total_area == 0:
                symm_score = 0
            else:
                area_diff = np.abs(M_left["m00"] - M_right["m00"])
                symm_score = (area_diff/ total_area) #divide by total_area

            if not looping:
                print(f"Angle {theta_deg}: left_area={M_left['m00']:.1f}, right_area={M_right['m00']:.1f}, score={symm_score:.3f}")
            
            #if symm_score > symm_cut_off:, is used if you want to only store the important parts
            lines_symm.append(theta_deg)
            scores.append(symm_score)
            moments_of_interest.append([M_left, M_right])

        else:
            left_cx, left_cy = IM.Centroid(left_cropped)
            right_cx, right_cy = IM.Centroid(right_cropped)

            M_L = IM.Hu_Moments(left_cropped, left_cx, left_cy)
            M_R = IM.Hu_Moments(right_cropped, right_cx, right_cy)

            #constant used to avoid division by zero
            const = 1e-15

            M_L = -np.sign(M_L) * np.log10(abs(M_L) + const)
            M_R = -np.sign(M_R) * np.log10(abs(M_R) + const)

            #M_L = np.array([- math.copysign(1, M_L[0]) * np.log10(abs(M_L[0]) + const),- math.copysign(1, M_L[1]) * np.log10(abs(M_L[1]) + const),- math.copysign(1, M_L[2]) * np.log10(abs(M_L[2]) + const),- math.copysign(1, M_L[3]) * np.log10(abs(M_L[3]) + const),- math.copysign(1, M_L[4]) * np.log10(abs(M_L[4]) + const), - math.copysign(1, M_L[5]) * np.log10(abs(M_L[5]) + const), - math.copysign(1, M_L[6]) * np.log10(abs(M_L[6]) + const)])

            #M_R = np.array([- math.copysign(1, M_R[0]) * np.log10(abs(M_R[0]) + const),- math.copysign(1, M_R[1]) * np.log10(abs(M_R[1]) + const),- math.copysign(1, M_R[2]) * np.log10(abs(M_R[2]) + const),- math.copysign(1, M_R[3]) * np.log10(abs(M_R[3]) + const),- math.copysign(1, M_R[4]) * np.log10(abs(M_R[4]) + const),- math.copysign(1, M_R[5]) * np.log10(abs(M_R[5]) + const),- math.copysign(1, M_R[6]) * np.log10(abs(M_R[6]) + const)])

            vec_total = np.abs(M_L) + np.abs(M_R) 

            #an array used for checking if the vec_total is zero
            empty_array = np.zeros(7)
            
            if np.equal(vec_total, empty_array).all():
                symm_score = 0

            else:
                #constant used to avoid division by zero
                vec_total += const
                
                vec_diff = np.abs(M_L - M_R)

                symm_score = vec_diff / vec_total #/ vect_total
                
            if not looping:
                print(f"Angle {theta_deg}: left Hu Moment={M_L:.1f}, left Hu Moment={M_R:.1f}, score={symm_score:.3f}")
            
            #if symm_score > symm_cut_off:, is used if you only want to store the important parts
            lines_symm.append(theta_deg)
            scores.append(symm_score)
            moments_of_interest.append([M_L, M_R])

    #cv.destroyAllWindows()
    return lines_symm, scores, moments_of_interest