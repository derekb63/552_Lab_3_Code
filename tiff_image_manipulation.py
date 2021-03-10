import matplotlib.pyplot as plt
import os
import numpy as np
import cv2


def find_tif_files(path):
    """Search in the path for all files that end with .tiff or .tif

       Parameters
       ----------
       path : str
           The location of the folder to look through.

       Returns
       -------
       tif_files : list
           a list of all the files found that match the criteria
       """
    tif_files = []
    for root, dirnames, filenames in os.walk(path):
        for file in filenames:
            if file.endswith(('.tif', '.tiff')):
                tif_files.append(os.path.join(root, file))
    return tif_files


def convert_to_int(string_to_convert):
    try:
        return [int(x) for x in string_to_convert.split(',')]
    except ValueError:
        return [round(float(x)) for x in string_to_convert.split(',')]


def calibration(calibration_image_path, cal_length):
    """Determine the pixel pitch of the image in meters per pixel

       Parameters
       ----------
       calibration_image_path : str or list
           The location of the calibration image(s).
       cal_length: float
           The length of the calibration object in m.

       Returns
       -------
       meters_per_pixel : float
           The calculated pixel pitch in meters per pixel.
       point_one: list
           The location chosen of the first edge of the calibration object.
       point_two: list
           The location chosen of the second edge of the calibration object.
       """
    if type(calibration_image_path) is str:
        # calibration_image = Image.open(calibration_image_path)
        calibration_image = cv2.imread(calibration_image_path, cv2.IMREAD_ANYDEPTH)
    elif type(calibration_image_path) is list:
        calibration_image = average_images(calibration_image_path)
    else:
        calibration_image = calibration_image_path
    plt.imshow(calibration_image)
    plt.show(block=True)
    point_one = input('Input the pixel location of the first side of the calibration object xxxx, yyyy:')
    point_one = convert_to_int(point_one)
    point_two = input('Input the pixel location of the second side of the calibration object xxxx, yyyy:')
    point_two = convert_to_int(point_two)
    meters_per_pixel = float(cal_length)/abs(float(point_one[0]) - float(point_two[0]))
    return meters_per_pixel, point_one, point_two


def hot_spot_points(hot_spot_image, num_points=4):
    """Collect the points of the hot spots in the images

       Parameters
       ----------
       hot_spot_image : str or list
           The location of the image(s) to pick the points from.
       num_points : int
           Number of points to collect

       Returns
       -------
       point_list : list
           List of x and y points collected
       """
    if type(hot_spot_image) is str:
        # hs_image = Image.open(hot_spot_image)
        hs_image = cv2.imread(hot_spot_image, cv2.IMREAD_ANYDEPTH)
    elif type(hot_spot_image) is list:
        hs_image = average_images(hot_spot_image)
    else:
        hs_image = hot_spot_image
    plt.figure('Figure for Determining Hot Spot Images')
    plt.imshow(hs_image)
    plt.show(block=True)
    point_list = []
    for i in range(num_points):
        point = input('Input the pixel location you wish to record xxxx, yyyy:')
        point = convert_to_int(point)
        point_list.append(point)
    return point_list


def average_images(list_of_paths):
    """Average a list of images into a single image

       Parameters
       ----------
       list_of_paths: list
           List of the images to be averaged.

       Returns
       -------
        : PIL.Image.Image
           The averaged image that has been converted to grayscale
       """
    w, h = cv2.imread(list_of_paths[0], cv2.IMREAD_ANYDEPTH).shape
    n = len(list_of_paths)
    average_array = np.zeros((h, w), np.float64)
    for image in list_of_paths:
        average_array += cv2.imread(image, cv2.IMREAD_ANYDEPTH)
    return np.divide(average_array, n)


def flame_slope_points(point_1, point_2, image_height, image_width):
    """Determine the fit of a line between the two points and return points on the line that are within the
       bounds of the image.

       Parameters
       ----------
       point_1: list
           Two item list locating the first point.
       point_2: list
           Two item list locating the second point.
       image_height: int
           Pixel height of the image.
       image_width: int
           Pixel width of the image.

       Returns
       -------
        fit_x_values: list
           List of the x pixel locations of the fit
        fit_y_values: list
           List of the y pixel locations of the fit
       """
    fit = np.polyfit((point_1[0], point_2[0]), (point_1[1], point_2[1]), deg=1)
    height_fit = np.poly1d(fit)
    x_values = np.array(range(int(image_width)))
    fit_values = height_fit(x_values)
    potential_indexes = (fit_values < image_height) & (fit_values > 0)
    fit_y_values = np.round(fit_values[potential_indexes]).astype(int)
    fit_x_values = x_values[potential_indexes].astype(int)
    return fit_x_values, fit_y_values


def line_intersection(line_1_points, line_2_points):
    """Determine the intersection point of two lines as described by:
       https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

       Parameters
       ----------
       line_1_points: list
           List of points that lie on the first line of the form [[x1, y1], [x2, y2]].
       line_2_points: list
           List of points that line on the second line [[x1, y1], [x2, y2]].

       Returns
       -------
       intersection_x: float
           x location of the intersection
       intersection_y: float
           y location of the intersection
       """
    x1, y1 = line_1_points[0]
    x2, y2 = line_1_points[1]
    x3, y3 = line_2_points[0]
    x4, y4 = line_2_points[1]

    determinant = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    intersection_x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1-x2)*(x3*y4 - y3*x4))/determinant
    intersection_y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1-y2)*(x3*y4 - y3*x4))/determinant
    return intersection_x, intersection_y


def calibration_figure(cal_point_left, cal_point_right, cal_object_length, cal_image, save_cal_fig=False):
    """Create the calibration plot for the report with annotated dimensions

       Parameters
       ----------
       cal_point_left: list
           [x, y] for the left point of the calibration
       cal_point_right: list
           [x, y] for the right point of the calibration
       cal_object_length: float
           length of the calibration object in meters
       cal_image: PIL.Image.Image
           image of the calibration object
       save_cal_fig: bool
           boolean to save the image. the image may also be saved in the figure pop-up window

       Returns
       -------
       None
       """
    plt.figure('Calibration Image for Report')
    plt.imshow(cal_image)
    calibration_pixels = abs(cal_point_left[0] - cal_point_right[0])
    plt.plot((cal_point_left[0], cal_point_right[0]), [(cal_point_left[1] + cal_point_right[1]) / 2] * 2, '-o')
    plt.annotate('Number of pixels (n):{0} \n Calibration Length (L):{1}m'.format(calibration_pixels,
                                                                                  cal_object_length),
                 [512, 512], ha='center', bbox=dict(boxstyle="round", fc="0.8"))
    plt.show()
    if save_cal_fig is True:
        plt.savefig('calibration_figure.png')
    return None


def cone_lateral_area(radius, h):
    """Determine the lateral area of a right circular cone

       Parameters
       ----------
       radius: float
           Radius of the cone base
       h: float
           Height of the cone

       Returns
       -------
       : float
           lateral area of the cone
       """
    return np.pi * radius * np.sqrt(h**2 + radius**2)


def annotate_dim(xy_from, xy_to, text):
    """create a dimension arrow on the current matplotlib plot

       Parameters
       ----------
       xy_from: list
           x and y location to start the arrow at
       xy_to: list
           x and y location to end the arrow
       text: str
          Text for the arrow label

       Returns
       -------
       none
       """
    plt.annotate("", xy_from, xy_to, arrowprops=dict(arrowstyle='<->', color='w', connectionstyle="bar,fraction=-0.3"))
    plt.text(xy_to[0] + 0.05 * xy_to[0], (xy_to[1] + xy_from[1]) / 2, text, color='w', ha='left', va='top')
    return None


if __name__ == '__main__':
    """Code to conduct the image processing for the ME552 Lab 3 ICCD camera images. You will have to run this code twice
    to get meaningful values. The first time will be to get the calibration values for the flame location and the second
    will be to generate the actual values.  You will want to rename the files to have exactly one of the following
    
        calibration: denotes an image to be used for calibration
        background:  denotes an image to be used for the background subtraction
        flame:       denotes an image that has a flame in it to be used for measurements
        
    The file names can include any other words or numbers but must have exactly one of the above words
    """
    # Edit the values in this block. You should not need to edit anything else
    image_directory = '/home/derek/Documents/552_Lab_Data/Example_Lab_3/'
    calibration_length = 0.0389128
    # End of the edit values block

    image_files = find_tif_files(image_directory)  # get all of the filepaths to the images
    # Sort the images by named type
    calibration_files = [image_files.pop(idx) for idx, img in enumerate(image_files) if 'calibration' in img]
    background_images = [image_files.pop(idx) for idx, img in enumerate(image_files) if 'background' in img]
    image_files = [image_files.pop(idx) for idx, img in enumerate(image_files) if 'flame' in img]
    # Get the necessary calibration values
    cal_val, cal_left, cal_right = calibration(calibration_files, calibration_length)
    # Generate the calibration figure to include the values needed for the report
    calibration_figure(cal_left, cal_right, calibration_length, average_images(calibration_files))

    # Load and average the background images
    average_background = average_images(background_images)
    # Load and average the flame images
    flame = average_images(image_files)
    # Subtract the averaged background from the flame images and convert to an array
    # subtracted = ImageChops.subtract(flame, average_background)
    subtracted = cv2.subtract(flame, average_background)
    subtracted = np.asarray(subtracted)
    subtracted = np.where(subtracted < 0, 0, subtracted)
    # Get the height and width of the images
    height, width = subtracted.shape
    # Collect the points of the hot spots to calculate the area of the flame
    hot_spot_locations = hot_spot_points(subtracted, 4)
    left_hot_spot_points = hot_spot_locations[:2]
    right_hot_spot_points = hot_spot_locations[2:]

    # Generate the linear curve fits that determine the edge of the flames
    fit_x_left, fit_y_left = flame_slope_points(left_hot_spot_points[0], left_hot_spot_points[1], height, width)
    fit_x_right, fit_y_right = flame_slope_points(right_hot_spot_points[0], right_hot_spot_points[1], height, width)
    # Calculate the location of the base of the flame as the first point alone the fit lines with a non-zero intensity
    flame_base_left = np.where(subtracted[fit_y_left, fit_x_left] > 0)[0]
    flame_base_left = [fit_x_left[flame_base_left[0]], fit_y_left[flame_base_left[0]]]
    flame_base_right = np.where(subtracted[fit_y_right, fit_x_right] > 0)[0]
    # Note that this uses the last value as compared to the first since this fit is sloped in the -x direction
    flame_base_right = [fit_x_right[flame_base_right[-1]], fit_y_right[flame_base_right[-1]]]
    # Take the flame tip as the points where the linear fits of the sides meet
    flame_tip = line_intersection(left_hot_spot_points, right_hot_spot_points)
    # Calculate the width and height of the flame
    flame_width = abs(flame_base_right[0] - flame_base_left[0]) * cal_val
    flame_height = abs(flame_tip[1] - flame_base_left[1]) * cal_val
    """Determine the center of the flame and calculate the tilt or skewness angle of the flame. Note that for the area
    calculation the equation for a right cone is used instead of one for the oblique cone. This is probably close 
    enough for low skewness but would be important to consider for high angles and for published research"""
    flame_base_center = (flame_base_left[0] + abs(flame_base_left[0] - flame_base_right[0])/2) * cal_val
    skewness_angle = 90 - (np.arctan(flame_height/abs(flame_base_center-flame_tip[0]*cal_val)) * (180/np.pi))
    flame_area = cone_lateral_area(flame_width/2, flame_height)

    # Create a figure that is annotated with all of the requirements for the lab report
    plt.figure('Averaged and Background Subtracted Flame Image')
    plt.imshow(subtracted, cmap='gray')
    plt.plot(fit_x_left, fit_y_left, label='Flame Left Edge')
    plt.plot(fit_x_right, fit_y_right, label='Flame Right Edge')
    plt.plot([flame_base_left[0], flame_base_right[0]], [flame_base_right[1]]*2, label='Flame Base')
    plt.plot(flame_base_left[0], flame_base_left[1], 'o', label='Flame Anchor Left')
    plt.plot(flame_base_right[0], flame_base_right[1], 'o', label='Flame Anchor Right')
    plt.plot(flame_tip[0], flame_tip[1], 'o', label='Flame Tip')
    annotate_dim(flame_base_left, flame_base_right, 'Flame width = {:.6f}m'.format(flame_width))
    annotate_dim(flame_base_right, (flame_base_right[0], flame_tip[1]), 'Flame Height = {:.6f}m'.format(flame_height))
    plt.legend()
    plt.show()

    # Print print out the necessary results
    print()
    print()
    print('The calibration result is {:.6f} meters per pixel'.format(cal_val))
    print('The flame height is {:.6f} meters'.format(flame_height))
    print('The flame area is {:.6f} square meters'.format(flame_area))
    print('The flame tilt is {:.6f} degrees'.format(skewness_angle))
