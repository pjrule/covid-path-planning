import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import triangulate
from shapely.geometry import box, Polygon, Point, LineString, MultiPolygon, MultiPoint
from room import Room
from tqdm import tqdm
from collections import deque


def poly_sightlines(room, exterior_coords):
    sightlines = np.zeros((len(exterior_coords[0]), room.guard_grid.shape[0]))
    for ext_idx, (x, y) in enumerate(zip(*exterior_coords)):
        for guard_idx, guard_point in enumerate(room.guard_grid):
            sight = LineString([guard_point, (x, y)])
            if room.room.contains(sight):
                sightlines[ext_idx, guard_idx] = 1
    return sightlines


def poly_intensities(room, weights, height, exterior_coords, wall_mask=None):
    box_x, box_y = exterior_coords
    gx = room.guard_grid[:, 0]
    gy = room.guard_grid[:, 1]
    intensities = np.zeros((len(box_x), room.guard_grid.shape[0]))
    for idx, (x, y) in enumerate(zip(box_x, box_y)):
        distance_2d = np.sqrt((x - gx) ** 2 + (y - gy) ** 2)
        if wall_mask is not None and wall_mask[idx]:
            angle = np.arctan(height / distance_2d)
        else:
            angle = np.pi / 2 - np.arctan(height / distance_2d)
        intensities[idx] = np.cos(angle) * weights / (4 * np.pi * (distance_2d**2 + height**2))
        #intensities[idx] = np.cos(angle) * weights / (4 * np.pi * (distance_2d**2 + height**2))
    return intensities


def boundary_mask(room, exterior_coords):
    return np.array([1 if room.room.boundary.contains(Point(x, y)) else 0
                     for (x, y) in zip(exterior_coords[0], exterior_coords[1])])


def illumination_lower_bound(room, tri, weights, height, robot_radius=None, shadow=False):
    coords = tri.exterior.coords.xy
    coords = (coords[0][:-1], coords[1][:-1])
    wall_mask = boundary_mask(room, coords)
    floor_intensities = poly_intensities(room, weights, height, coords)
    wall_intensities = poly_intensities(room, weights, height, coords, wall_mask)
    intensities = np.minimum(floor_intensities, wall_intensities)
    sightlines = poly_sightlines(room, coords)
    strong_sightlines = np.tile(np.bitwise_and.reduce(sightlines.astype(int), 0), (3, 1))
    intensities[strong_sightlines == 0] = np.inf
    min_intensities = np.min(intensities, axis=0)
    min_intensities[min_intensities == np.inf] = 0
    return np.sum(min_intensities)

def illumination_upper_bound(room, tri, weights, height):
    p_x = tri.centroid.coords.xy[0][0]
    p_y = tri.centroid.coords.xy[1][0]
    gx = room.guard_grid[:, 0]
    gy = room.guard_grid[:, 1]
    intensities = np.zeros(room.guard_grid.shape[0])
    distance_2d = (p_x - gx)**2 + (p_y - gy)**2
    angle = np.arctan(height / distance_2d)
    for idx, (x, y) in enumerate(zip(gx, gy)):
        sight = LineString([(p_x, p_y), (x, y)])
        if room.room.contains(sight):
            intensities[idx] = (
                max(np.cos(angle[idx]), np.cos((np.pi / 2) - angle[idx])) *
                weights[idx] / (4 * np.pi * (distance_2d[idx] + height**2))
            )
    return np.sum(intensities)


def split_triangle(room, tri):
    coords = tri.exterior.coords.xy
    points = [Point(x, y) for x, y in zip(*coords)]
    points.append(tri.centroid)
    return triangulate(MultiPoint(points))
      
def find_ear(poly):
    coords = poly.exterior.coords
    n = len(coords.xy[0])
    for idx in range(n):
        p_1 = coords[(idx - 1) % n]
        p_2 = coords[idx % n]
        p_3 = coords[(idx + 1) % n]
        diag = LineString([p_1, p_3])
        diag_inter = diag.intersection(poly)
        principal_vertex = isinstance(diag_inter, LineString) and len(diag_inter.coords) == 2
        ear = poly.contains(diag)
        if principal_vertex and ear:
            return Polygon([p_1, p_2, p_3, p_1])


def ear_triangulate(poly):
    tris = []
    orig_poly = poly

    while len(poly.exterior.coords.xy[0]) > 4:
        fig, ax = plt.subplots()
        ax.plot(*orig_poly.exterior.coords.xy, color='red', linestyle='--', linewidth=2)
        for tri in tris:
            ax.plot(*tri.exterior.coords.xy)
        plt.show()

        next_tri = find_ear(poly)
        tris.append(next_tri)
        poly = poly.difference(next_tri).buffer(0)
    return tris + [poly]


def branch_bound_poly(room,
                     weights,
                     robot_height,
                     max_iters=15,
                     epsilon=1e-5,
                     shadow=False,
                     robot_radius=None):
    # Step 1: triangulate the polygon.
    exterior = room.room
    if exterior and isinstance(exterior, MultiPolygon):
        tris = []
        for poly in exterior:
            tris += ear_triangulate(poly)
    else:
        tris = ear_triangulate(exterior)

    ear_triangulate(room.room)

    fig, ax = plt.subplots()
    ax.plot(*room.room.exterior.coords.xy, color='red', linestyle='--', linewidth=2)
    for t in tris:
        ax.plot(*t.exterior.coords.xy)
    plt.scatter(room.guard_grid[:, 0], room.guard_grid[:, 1])
    plt.show()

    # Step 2: Split the triangles until the strong visibility condition
    # holds at each triangle: each vertex is seen by the same
    # guard point for at least one guard point.
    valid_tris = []
    tri_queue = deque(t for t in tris)
    while tri_queue:
        tri = tri_queue.popleft()
        coords = tri.exterior.coords.xy
        coords = (coords[0][:-1], coords[1][:-1])
        sightlines = poly_sightlines(room, coords)
        if np.max(np.sum(sightlines, axis=0)) < 3:
            tri_queue += split_triangle(room, tri)
        else:
            valid_tris.append(tri)

    # Step 3: triangulate and bound. We use illumination at the centroid
    # of each triangulate as an upper bound on minimum illumination; we
    # use strong visibility to find a lower bound.
    lbs = [
        illumination_lower_bound(room, tri, weights, robot_height, robot_radius, shadow)
        for tri in valid_tris
    ]
    ubs = [
        illumination_upper_bound(room, tri, weights, robot_height)
        for tri in valid_tris
    ]
    L_idx = 0
    Q = valid_tris[0]
    L = lbs[0]
    U = ubs[0]
    
    for i in range(max_iters):
        fig, ax = plt.subplots()
        #ax.plot(*room.room.exterior.coords.xy, color='red', linestyle='--', linewidth=3)
        for tri in valid_tris:
            ax.plot(*tri.exterior.coords.xy, color='red')
        plt.scatter(room.guard_grid[:, 0], room.guard_grid[:, 1])
        plt.show()

        if i > 0:
            print(i, L, U, U - L)
        if U - L <= epsilon or Q.area < 1e-20:
            break

        new_tris = split_triangle(room, Q)
        valid_tris[L_idx] = new_tris[0]
        lbs[L_idx] = illumination_lower_bound(
            room, new_tris[0], weights, robot_height, robot_radius, shadow
        ) 
        ubs[L_idx] = illumination_upper_bound(
            room, new_tris[0], weights, robot_height
        )
        for tri in new_tris[1:]:
            valid_tris.append(tri)
            lbs.append(illumination_lower_bound( 
                room, tri, weights, robot_height, robot_radius, shadow
            ))
            ubs.append(illumination_upper_bound( 
                room, tri, weights, robot_height
            ))

        # Update bounds.
        L_idx = np.argmin(lbs)
        Q = valid_tris[L_idx]
        L = np.min(lbs)
        U = np.min(ubs)
    return Q, L, U
