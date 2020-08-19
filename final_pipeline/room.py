import cvxpy as cp
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import box, Point, LineString, Polygon, MultiPolygon
from shapely.affinity import scale
from scipy.spatial import distance_matrix
from matplotlib.animation import FuncAnimation


class Room:
    """Represents the geometries of a room and its guarded region.

    units_per_pixel: conversion between units of robot height and pixels
    room_eps, guard_eps: the max distance of any point in the room from a
                    point on the (room/guard) grid in robot-height units

    TODO: input geojson instead of polygon (ie. revert to the working code)
    """
    def __init__(self, polygon, robot_buffer_meters = 0, is_valid_guard = lambda x, y: True, room_eps=0.5, guard_eps=0.5):
        self.room_eps = room_eps
        self.guard_eps = guard_eps
        self.room = polygon

        print("getting guard")
        self.guard = self.room.buffer(-robot_buffer_meters)
        if self.guard.geom_type == 'MultiPolygon':
            self.guard = max(self.guard, key = lambda p: p.area)

        
        # plt.plot(*self.room.exterior.xy)
        # plt.plot(*self.guard.exterior.xy)
        # plt.show()

        print("getting room grid")
        self.room_grid, self.room_cells = self._grid(self.room, room_eps)

        print("getting guard grid")
        self.guard_grid, self.guard_cells = self._grid(self.guard, guard_eps, is_valid_guard)
        
        
    def _grid(self, geom, epsilon, is_valid = lambda x, y: True):
        """Returns points within a geometry (gridded over its bounding box).
        
        Points on the grid inside the bounding box but outside the geometry
        are rejected.
        
        :param epsilon: The length of a grid cell side
        """
        minx, miny, maxx, maxy = geom.bounds

        x_arr = np.arange(minx, maxx, epsilon)
        y_arr = np.arange(miny, maxy, epsilon)
        xx, yy = np.meshgrid(x_arr, y_arr)
        filtered_points = []
        filtered_cells = []
        for x, y in zip(xx.flatten(), yy.flatten()):
            is_in_geom, data = self._get_grid_cell(x, y, epsilon, geom)
            if is_in_geom and is_valid(x,y):
                cells, cell_points = data
                filtered_points.extend([(point.x, point.y) for point in cell_points])
                filtered_cells.extend(cells)

        return np.array(filtered_points), np.array(filtered_cells)
    
    def _get_grid_cell(self, x, y, box_size, geom):
        """Computes a grid cell, the intersection of geom and rectangle centered on (x, y)

        Returns a boolean indicating if the grid cell is empty and a data object.
        If the grid cell is not empty, `data` is tuple that contains
        a list of simple polygons (shapely.Polygon) that compose the intersection
        and a list of representatives points (shapely.Point) inside the polygons
            
        Throws an error if the grid cell is not a simple polygon.
        """
        minx = x - box_size/2
        maxx = x + box_size/2
        miny = y - box_size/2
        maxy = y + box_size/2
        unfiltered_cell = box(minx = minx, miny = miny, maxx = maxx, maxy = maxy)
        intersection = geom.intersection(unfiltered_cell)
        if intersection.is_empty:
            is_in_geom = False
            data = None
        elif isinstance(intersection, Polygon):
            assert intersection.is_simple, "Increase grid resolution to ensure grid cells are simple polygons"
            is_in_geom = True
            cells = [intersection]
            cell_points = [intersection.representative_point()]
            data = (cells, cell_points)
        elif isinstance(intersection, MultiPolygon):
            is_in_geom = True
            cells = list(intersection)
            cell_points = [cell.representative_point() for cell in cells]
            data = (cells, cell_points)
        else:
            # This should never happen...
            assert(False)
        return is_in_geom, data
