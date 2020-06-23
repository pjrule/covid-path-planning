import cvxpy as cp
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import box, Point, LineString, Polygon, MultiPolygon
from shapely.affinity import scale
from scipy.spatial import distance_matrix
from matplotlib.animation import FuncAnimation
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

class Room:
    """Represents the geometries of a room and its guarded region."""
    def __init__(self, filename, room_res=1000, guard_res=1000, guard_scale=1):
        self.gdf = gpd.read_file(filename)
        self.guard_scale = guard_scale
        self.room = self.gdf[self.gdf['type'] == 'room'].iloc[0].geometry
        self.guard = box(*(scale(self.room, guard_scale, guard_scale).bounds))
        for obs in self.gdf[self.gdf['type'] == 'obstacle'].geometry:
            self.guard = self.guard.difference(obs)
        self.guard = self.guard.intersection(self.room)
        self.room_grid, self.room_cells, self.room_epsilon = self._grid(self.room, room_res)
        self.guard_grid, self.guard_cells, self.guard_epsilon = self._grid(self.guard, guard_res)
        
    @property
    def guard_geodesic_center(self):
        """Finds the best guard grid approximation of the room grid's geodesic center."""
        # The geodesic center minimizes the maximum distance to any point.
        dist = distance_matrix(self.guard_grid, self.room_grid)
        return np.argmin(np.max(dist, axis=1))
        
    def _grid(self, geom, res):
        """Returns points within a geometry (gridded over its bounding box).
        
        Points on the grid inside the bounding box but outside the geometry
        are rejected.
        
        :param res: The number of points in the bounding box's grid (approx.)
        """
        minx, miny, maxx, maxy = geom.bounds
        aspect = (maxy - miny) / (maxx - minx)
        n_x_points = int(np.ceil(np.sqrt(res / aspect)))
        n_y_points = int(np.ceil(np.sqrt(res)))
        x_arr, x_epsilon = np.linspace(minx, maxx, n_x_points, retstep = True)
        y_arr, y_epsilon = np.linspace(miny, maxy, n_y_points, retstep = True)
        xx, yy = np.meshgrid(x_arr, y_arr)
        filtered_points = []
        filtered_cells = []
        for x, y in zip(xx.flatten(), yy.flatten()):
            is_in_geom, data = self._get_grid_cell(x, y, x_epsilon, y_epsilon, geom)
            if is_in_geom:
                cells, cell_points = data
                filtered_points.extend([(point.x, point.y) for point in cell_points])
                filtered_cells.extend(cells)
                
        # Every point in the room is within epsilon of a point in the grid
        grid_epsilon = np.sqrt(x_epsilon**2 + y_epsilon**2)
        
        return np.array(filtered_points), np.array(filtered_cells), grid_epsilon
    
    def _get_grid_cell(self, x, y, x_epsilon, y_epsilon, geom):
        """Computes a grid cell, the intersection of geom and rectangle centered on (x, y)

        Returns a boolean indicating if the grid cell is empty and a data object.
        If the grid cell is not empty, `data` is tuple that contains
        a list of simple polygons (shapely.Polygon) that compose the interseciton
        and a list of representatives points (shapely.Point) inside the polygons
            
        Throws an error if the grid cell is not a simple polygon.
        """
        minx = x - x_epsilon/2
        maxx = x + x_epsilon/2
        miny = y - y_epsilon/2
        maxy = y + y_epsilon/2
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
        
