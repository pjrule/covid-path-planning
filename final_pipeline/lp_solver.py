import cvxpy as cp
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import box, Point, LineString, Polygon, MultiPolygon
import shapely.geometry
from shapely.affinity import scale
from shapely.prepared import prep
from scipy.spatial import distance_matrix
from matplotlib.animation import FuncAnimation
from numpy.linalg import norm
from sys import exit
from shapely.ops import transform

# Try to load skgeom package (for naive solution)
SKGEOM_AVAIL = False
try:
    from skgeom import *
    SKGEOM_AVAIL = True
except ImportError as e:
    pass

from branch_and_bound import branch_bound_poly

EPS = 1e-5 # Arbitrary small number to avoid rounding error

def solve_full_lp(room, robot_height, robot_radius, robot_power, use_strong_visibility, use_strong_distances, scaling_method, min_intensity):
    room_intensities = get_intensities(room, robot_height, robot_radius, robot_power, use_strong_visibility, use_strong_distances)
    loc_times = cp.Variable(room.guard_grid.shape[0])
    obj = cp.Minimize(cp.sum(loc_times))
    constraints = [
        room_intensities.T @ loc_times >= min_intensity,
        loc_times >= 0
    ]
    prob = cp.Problem(obj, constraints=constraints)
    prob.solve(solver='ECOS')

    # Report if problem is infeasible
    if prob.status == 'infeasible':
        exit('ERROR: Problem is infeasible')
    
    unscaled_time = loc_times.value.sum()
    scale = get_scale(room, loc_times.value, scaling_method)
    solution_time = scale * unscaled_time

    # TODO: Filter for significant points

    return solution_time, loc_times.value, room_intensities.T

def visualize_times(room, waiting_times):
    ax = plt.axes()
    ax.axis('equal')
    ax.imshow(room.room_img)
    ax.plot(*transform(room.xy_to_pixel, room.room).exterior.xy)

    guard_points_x = [room.xy_to_pixel(x, y)[0] for x, y in room.guard_grid]
    guard_points_y = [room.xy_to_pixel(x, y)[1] for x, y in room.guard_grid]
    ax.scatter(guard_points_x, guard_points_y, s = waiting_times / 10, facecolors = 'none', edgecolors = 'r', alpha = 0.5)
    plt.show()


# Intensity is power transmitted per unit area
def get_intensities(room, robot_height, robot_radius, robot_power, use_strong_visibility = True, use_strong_distances = True):
    # Construct initial intensities matrix, ignoring visibility
    # *Do* account for inverse distance, angle, and robot shadow
    num_guard_points = room.guard_grid.shape[0]
    num_room_points = room.room_grid.shape[0]
    room_intensities = np.zeros(shape = (num_guard_points, num_room_points))

    for guard_idx, guard_pt in enumerate(room.guard_grid):
        if guard_idx % 50 == 0: print("Getting intensity for guard point: {}/{}".format(guard_idx, room.guard_grid.shape[0]))
        for room_idx, room_pt in enumerate(room.room_grid):
            if use_strong_distances:
                room_cell = room.room_cells[room_idx]
                distance_2d = Point(guard_pt).hausdorff_distance(room_cell)
                dist_for_shadow = Point(guard_pt).distance(room_cell) # Shortest distance: worst case scenario
            else:
                distance_2d = norm(guard_pt - room_pt)
                dist_for_shadow = distance2d

            angle = np.pi/2 - np.arctan(robot_height/distance_2d)

            # Account for robot shadow
            if dist_for_shadow < robot_radius:
                room_intensities[guard_idx, room_idx] = 0
            else:
                # Energy received follows an inverse-square law and Lambert's cosine law
                room_intensities[guard_idx, room_idx] = robot_power * np.cos(angle) * 1/(4 * np.pi * (distance_2d**2 + robot_height**2))

    # Patch up visibility.
    eps_room = prep(room.room.buffer(EPS))
    broken_sightlines_count = 0
    broken_sightlines_list = []
    not_visible_room_idx = []
    for room_idx, room_point in enumerate(room.room_grid):
        if room_idx % 50 == 0: print("Processing room index: {}/{}".format(room_idx, room.room_grid.shape[0]))
        none_visible = True
        for guard_idx, guard_point in enumerate(room.guard_grid):
            if use_strong_visibility:
                # If a point can see all vertices of a simple polygon,
                # then it can see the entire polygon
                room_cell = room.room_cells[room_idx]
                room_cell_points = list(room_cell.exterior.coords)
                sightlines = [LineString([pt, guard_point]) for pt in room_cell_points]
                is_visible = all([eps_room.contains(line) for line in sightlines])
            else:
                sight = LineString([guard_point, room_point])
                is_visible = eps_room.contains(sight)
            if not is_visible:
                broken_sightlines_list.append((guard_point, room_point))
                broken_sightlines_count += 1
                room_intensities[guard_idx, room_idx] = 0
            else:
                none_visible = False

        if none_visible:
            # Ignore this room point - do not guarantee that it is illuminated
            room_intensities[:, room_idx] = 1
            not_visible_room_idx.append(room_idx)
            print('Unreachable Room Point:', room_point)

    if len(not_visible_room_idx) > 0:
        print('CAUTION: Not all points in the room can be illuminated')
        print('         Red regions will not be disinfected by the robot')
        plt.imshow(room.room_img)
        for i in not_visible_room_idx:
            plt.fill(*transform(room.xy_to_pixel, room.room_cells[i]).exterior.coords.xy, 'red')
        plt.plot(*transform(room.xy_to_pixel, room.room).exterior.coords.xy, 'black')
        plt.show()

    print('Removed', broken_sightlines_count, 'broken sightlines')
    
    return room_intensities


def get_scale(room, waiting_times, scaling_method):
    if scaling_method == 'epsilon':
        print('ERROR: Epsilon scaling not currently supported. Need to convert epsilon to units of robot height')
        return None
        scale = (np.sqrt(room.room_eps**2 + 4) + room.room_eps)/(np.sqrt(room.room_eps**2 + 4) - room.room_eps)
    elif scaling_method == 'branch_and_bound':
        _, lower_bound, _ = branch_bound_poly(room, waiting_times, max_iters = 50)
        scale = min_intensity/lower_bound
    elif scaling_method == 'none':
        scale = 1
    else:
        scale = None
        print('Unrecognized scaling method:', scaling_method)

    print(scale)
    return scale

#############################
## Code for naive solution ##
#############################

# Naive solution:
#   Place a point near the center of the room and illuminate all parts of the room it can see
#   Repeat the process for the regions that have not yet been covered
def solve_naive(room, robot_height, min_intensity):
    if not SKGEOM_AVAIL:
        exit('Error: Naive solution is not available (skgeom library could not be loaded)')

    plt.close()
    covered_regions = None # Multipolygon
    solution_points = []
    solution_times = []

    not_covered = room.room
    time = 0

    while not not_covered.is_empty:
        # NB: I'm not completely sure how the representative_point() method works,
        #     but the scikit-geometry code will fail if this ever returns a vertex
        point = not_covered.representative_point()
        vis = compute_visibility_polygon(room.room, point).buffer(EPS)
        to_be_covered = not_covered.intersection(vis)

        not_covered = not_covered.difference(vis)

        max_distance = point.hausdorff_distance(to_be_covered)
        curr_time = min_intensity * (max_distance**2 + robot_height**2)
        time += curr_time
        solution_points.append(point)
        solution_times.append(curr_time)
        #plot_multi(room.room, to_be_covered, solution_points, 'orange')
        plot_multi(room.room, not_covered, solution_points)
    
    print('Total solution time:', time)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.plot(*room.room.exterior.xy)
    ax.scatter([pt.x for pt in solution_points], [pt.y for pt in solution_points], s = np.array(solution_times)/10, facecolors = 'none', edgecolors = 'r')
    plt.show()

    return time, solution_times

# Plot a shapely polygon or multipolygon with matplotlib
def plot_multi(room_polygon, multi, solution_points, color = 'r'):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    pointx = [point.x for point in solution_points]
    pointy = [point.y for point in solution_points]
    ax.scatter(pointx, pointy)
    ax.plot(*room_polygon.exterior.xy)

    if multi.is_empty:
        pass
    elif multi.geom_type == 'Polygon':
        ax.plot(*multi.exterior.xy, color = color)
    elif multi.geom_type == 'MultiPolygon':
        for poly in multi:
            ax.plot(*poly.exterior.xy, color = color)
    plt.show()

# Find the visibility polygon of a point
# Inputs and output are shapely objects
# Converts to scikit-geometry to find visibility intersection
# Note: this is inefficient for repeated calls on the same shapely polygon
def compute_visibility_polygon(polygon, point):
    points_list = list(polygon.exterior.coords) # Includes a repeating point at the end
    xy_segments = [(points_list[i], points_list[i+1]) for i in range(len(points_list) - 1)]
    segments = [Segment2(Point2(x1, y1), Point2(x2, y2)) for ((x1, y1), (x2, y2)) in xy_segments]

    arr = arrangement.Arrangement()
    for s in segments:
        arr.insert(s)

    vs = RotationalSweepVisibility(arr)

    pt = Point2(point.x, point.y)
    face = arr.find(pt)
    vx = vs.compute_visibility(pt, face)

    visibility_polygon_points = [(v.curve().point(0).x(), v.curve().point(0).y()) for v in vx.halfedges]
    return(shapely.geometry.Polygon(visibility_polygon_points))
