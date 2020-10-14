import cvxpy as cp
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import box, Point, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.affinity import scale
from scipy.spatial import distance_matrix
from matplotlib.animation import FuncAnimation
from shapely.ops import transform

class Room:
    """Represents the geometries of a room and its guarded region.

    units_per_pixel: conversion between units of robot height and pixels
    room_eps, guard_eps: the max distance of any point in the room from a
                    point on the (room/guard) grid in robot-height units

    TODO: input geojson instead of polygon (ie. revert to the working code)
    """
    def __init__(self, polygon, room_img, xy_to_pixel, robot_buffer_meters = 0, is_valid_guard = lambda x, y: True, room_eps=0.5, guard_eps=0.5):
        self.room_eps = room_eps
        self.guard_eps = guard_eps
        self.room = polygon
        self.room_img = room_img
        self.xy_to_pixel = xy_to_pixel

        self.guard = self.room.buffer(-robot_buffer_meters)
        if self.guard.geom_type == 'MultiPolygon':
            self.guard = max(self.guard, key = lambda p: p.area)

        self.room_grid, self.room_cells = self._grid(self.room, room_eps)
        self.guard_grid, self.guard_cells = self._grid(self.guard, guard_eps, is_valid_guard)
        self.wall_grid, self.wall_cells = self._grid(self.room.exterior, room_eps)

        self.full_room_grid = self.room_grid #np.append(self.room_grid, self.wall_grid, axis = 0)
        self.full_room_cells = self.room_cells #np.append(self.room_cells, self.wall_cells, axis = 0)
        self.full_room_iswall = np.full(self.room_cells.shape, False)#np.append(np.full(self.room_cells.shape, False),
                                #          np.full(self.room_cells.shape, True),
                                #          axis = 0)

        # Visualize all possible robot locations
        plt.imshow(self.room_img)
        #for guard_pt in self.guard_grid:
        #    plt.scatter(*self.xy_to_pixel(*guard_pt), color = 'blue')
        #plt.plot(*transform(self.xy_to_pixel, polygon).exterior.xy, color = 'red')
        for wall_cell in self.wall_cells:
            plt.plot(*transform(self.xy_to_pixel, wall_cell).xy)
        #for wall_pt in self.wall_grid:
        #    plt.scatter(*self.xy_to_pixel(*wall_pt), color = 'blue')
        plt.show()

        
        
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
            if is_in_geom:
                cells, cell_points = data
                filtered_points.extend([cell_points[i] for i in range(len(cell_points)) if is_valid(*cell_points[i])])
                filtered_cells.extend([cells[i]        for i in range(len(cell_points)) if is_valid(*cell_points[i])])

        return (np.asarray(filtered_points),
                np.asarray(filtered_cells, dtype = object))
    
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
        elif isinstance(intersection, Polygon) or isinstance(intersection, LineString):
            assert intersection.is_simple, "Increase grid resolution to ensure grid cells are simple polygons"
            is_in_geom = True
            cells = [intersection]
            cell_point_shapely = intersection.representative_point()
            cell_points = [(cell_point_shapely.x, cell_point_shapely.y)]
            data = (cells, cell_points)
        elif isinstance(intersection, MultiPolygon) or isinstance(intersection, MultiLineString):
            is_in_geom = True
            cells = list(intersection)
            cell_points_shapely = [cell.representative_point() for cell in cells]
            cell_points = [(point.x, point.y) for point in cell_points_shapely]
            data = (cells, cell_points)
        #elif isinstance(intersection, LineString):
        #    is_in_geom = True
        #    cells = [intersection]
        else:
            # This should never happen...
            raise Exception("Unable to classify intersection: ", intersection)

        return is_in_geom, data
