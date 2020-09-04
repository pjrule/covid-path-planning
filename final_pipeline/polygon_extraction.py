'''
polygon_extraction.py

Contains methods for the "preprocessing" step -  converting an input png into
a closed polygon representing the room
'''

import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import cv2
import math
from skimage.io import imread
from skimage.segmentation import watershed
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.affinity import scale
from shapely.ops import transform
from yaml import safe_load

from orthogonal_simplification import construct_orthogonal_polygon

# Extracts a polygon from a png of the room
# Input:
#   input_filepath:    str, the filepath to the input pgm
#   input_yaml:        str, a filepath to the input yaml describing the scan
#   contour_accuracy:  float passed to approxPolyDP
#                      This controls initial simplification done before the
#                      orthogonal approximation, used to speed up the runtime
#                      of the orthogonal approximation
#   ortho_tolerance:   float, the maximum distance the orthogonal approximation
#                      can deviate from the original curve
# Returns: a tuple containing
#   1. An orthogonal shapely Polygon approximating the room, using the
#      'meters-based' coordinate system defined by the yaml file
#   2. A 2D numpy array representing the room image stored in (row, col) order
#      where 0 = 'occupied', 205 = 'uknown', and 254 = 'unoccupied'
#   3. A method to convert (x, y) coordinates in the 'meters' space to (x, y)
#      coordinates in the 'image pixel' space.
#   4. A float describing meters per pixel
def extract_polygon(input_filepath, input_yaml, contour_accuracy = 2, ortho_tolerance = 20):
    raw = imread(input_filepath)
    gray = raw # If input path is png instead of pgm, set equal to: cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) (TODO: Automate)

    # https://stackoverflow.com/questions/35189322/opencv-polygon-detection-methods
    th, im_th = cv2.threshold(gray, 220, 250, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    im_th2 = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel)

    # Connected Components segmentation:
    maxLabels, labels = cv2.connectedComponents(im_th2)


    # We need to determine which label corresponds to the "background"
    #   connected component. There may be noise in the input data, but this
    #   should be the label present at all the edges of the image.
    # Heuristic: select the label that appears most frequently at the edges
    upper_edge_labels = labels[0, :]
    lower_edge_labels = labels[-1, :]
    left_edge_labels  = labels[:, 0]
    right_edge_labels = labels[:, -1]
    edge_labels = np.concatenate((upper_edge_labels, lower_edge_labels,
                                 left_edge_labels, right_edge_labels))
    background_label = np.bincount(edge_labels).argmax()


    room_outline = np.zeros_like(labels, dtype=np.uint8)
    room_outline[labels == background_label] = 1

    # Find the contour corresponding to the room outline
    # Ideally room_outline contains a single connected component (hence only
    # outline), but in practice there is some noise.
    # Heuristic: select the "longest" contour
    #            (the contour with the most points)
    contours, _ = cv2.findContours(room_outline,
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    room_contour_idx = np.argmax([contour.shape[0] for contour in contours])
    room_contour = contours[room_contour_idx]


    # Find approximate polygon
    poly = cv2.approxPolyDP(room_contour, contour_accuracy, True)
    p = poly.reshape(poly.shape[0], poly.shape[2])
    p = np.append(p, [p[0]], axis=0)
    p = Polygon(p)

    orthogonal_poly = construct_orthogonal_polygon(p, ortho_tolerance)


    # Display polygon with original image
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.imshow(gray)
    ax.plot(*orthogonal_poly.exterior.coords.xy, color='red', linewidth=2)
    ax.axis('equal')
    plt.show()


    # Parse yaml file
    with open(input_yaml, 'r') as yaml_file:
        scan_params = safe_load(yaml_file)
    
    meters_per_pixel = scan_params['resolution']
    lower_left_x = scan_params['origin'][0]
    lower_left_y = scan_params['origin'][1]

    # scikit-image interprets (0,0) as the top left corner
    # We want (0,0) to the be bottom left corner
    # To do this, flip around reflection line
    reflection_line_meters = gray.shape[0]/2 * meters_per_pixel

    # Convert from cv2.findContours coordinates to 'meters' coordinates
    def cv2_coords_to_xy(img_x, img_y):
        # Convert to meters
        x = img_x * meters_per_pixel
        y = img_y * meters_per_pixel

        # Flip y across center line
        y = -y + 2 * reflection_line_meters

        # Translate image so lower left corner, previously (0,0),
        # now has correct coordinates
        x = x + lower_left_x
        y = y + lower_left_y

        return (x, y)

    # Convert from 'meters' coordinates to cv2.findContours coordinates
    def xy_to_pixel(poly_x, poly_y):
        # Translate
        x = poly_x - lower_left_x
        y = poly_y - lower_left_y

        # Flip y across center line
        y = -y + 2 * reflection_line_meters

        # Convert to pixels
        x = x * 1/meters_per_pixel
        y = y * 1/meters_per_pixel

        return (int(x), int(y))

    orthogonal_poly = transform(cv2_coords_to_xy, orthogonal_poly)

    return(orthogonal_poly, gray, xy_to_pixel, meters_per_pixel)


# Returns a function to determine whether a given (x, y) coordinate in
# the 'meters' space is outside of a specified buffer around all
# occupied pixels in the gray room image
def construct_isValidLocation_function(gray_room_img, xy_to_pixel, robot_buffer, meters_per_pixel, avoid_unknown_regions):
    # Select the occupied areas the robot must avoid
    room_img = gray_room_img.copy()
    if avoid_unknown_regions:
        room_img[room_img == 205] = 0
    room_img[room_img != 0] = 1

    robot_diameter_pixels = math.ceil(2 * robot_buffer * 1/meters_per_pixel)
    # Force the kernel to have odd size so it is centered exactly on the location
    if robot_diameter_pixels % 2 == 0: robot_diameter_pixels += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(robot_diameter_pixels, robot_diameter_pixels))    
    expanded_walls = cv2.erode(room_img, kernel)

    def is_valid_location(x, y):
        img_col, img_row = xy_to_pixel(x, y)
        return expanded_walls[img_row, img_col] != 0

    return is_valid_location
