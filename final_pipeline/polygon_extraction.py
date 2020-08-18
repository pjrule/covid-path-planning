import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import cv2
from skimage.io import imread
from skimage.segmentation import watershed
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.affinity import scale

from orthogonal_simplification import construct_orthogonal_polygon

# Extracts a polygon from a png of the room
# Input:
#   input_png:         str, the filepath to the input png
#   output_png:        str, a filepath to save a png of the output polygon
#   contour_accuracy:  float passed to approxPolyDP
# Return: a shapely Polygon
def extract_polygon(input_filepath, output_filepath, contour_accuracy = 2, ortho_tolerance = 20):
    raw = imread(input_filepath)
    gray = raw # If input path is png instead of pgm, set equal to: cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) (TODO: Automate)

    # https://stackoverflow.com/questions/35189322/opencv-polygon-detection-methods
    th, im_th = cv2.threshold(gray, 220, 250, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    im_th2 = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel)

    #kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    #im_th3 = cv2.morphologyEx(im_th2, cv2.MORPH_OPEN, kernel2)

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

    fig, ax = plt.subplots(figsize=(4, 8))
    ax.imshow(raw)
    ax.plot(*orthogonal_poly.exterior.coords.xy, color='red', linewidth=2)
    plt.show()
    #plt.savefig(output_filepath)

    return(orthogonal_poly)
