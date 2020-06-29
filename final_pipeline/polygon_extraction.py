import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import cv2
from skimage.io import imread
from skimage.segmentation import watershed
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.affinity import scale

# Extracts a polygon from a png of the room
# Input:
#   input_png:         str, the filepath to the input png
#   output_png:        str, a filepath to save a png of the output polygon
#   units_per_pixel:   float by which to scale the output polygon
#   contour_accuracy:  float passed to approxPolyDP
# Return: a shapely Polygon
def extract_polygon(input_filepath, output_filepath, contour_accuracy = 5):
    raw = imread(input_filepath)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    # https://stackoverflow.com/questions/35189322/opencv-polygon-detection-methods
    th, im_th = cv2.threshold(gray, 220, 250, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    im_th2 = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)
    # Connected Components segmentation:
    maxLabels, labels = cv2.connectedComponents(im_th2)


    room_outline = np.zeros_like(labels, dtype=np.uint8)
    room_outline[labels == 1] = 1
    contours, _ = cv2.findContours(room_outline,
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)


    poly = cv2.approxPolyDP(contours[15], contour_accuracy, True)
    p = poly.reshape(poly.shape[0], poly.shape[2])
    p = np.append(p, [p[0]], axis=0)

    fig, ax = plt.subplots(figsize=(4, 8))
    ax.imshow(raw)
    ax.plot(p[:, 0], p[:, 1], color='red', linewidth=4)
    plt.savefig(output_filepath)

    return(Polygon(p))