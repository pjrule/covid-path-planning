import cvxpy as cp
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import box, Point, LineString, Polygon, MultiPolygon
from shapely.affinity import scale
from shapely.prepared import prep
from scipy.spatial import distance_matrix
from matplotlib.animation import FuncAnimation
from numpy.linalg import norm

from branch_and_bound import branch_bound_poly

EPS = 1e-5 # Arbitrary small number to avoid rounding error

def solve_full_lp(room, robot_height_scaled, use_strong_visibility, use_strong_distances, scaling_method, min_intensity):
    room_intensities = get_intensities(room, robot_height_scaled, use_strong_visibility, use_strong_distances)
    loc_times = cp.Variable(room.guard_grid.shape[0])
    obj = cp.Minimize(cp.sum(loc_times))
    constraints = [
        room_intensities.T @ loc_times >= min_intensity,
        loc_times >= 0
    ]
    prob = cp.Problem(obj, constraints=constraints)
    prob.solve(solver='ECOS')
    
    unscaled_time = loc_times.value.sum()
    scale = get_scale(room, loc_times.value, scaling_method)
    solution_time = scale * unscaled_time

    # TODO: Filter for significant points

    return solution_time, loc_times.value, room_intensities.T

def visualize_times(room, waiting_times):
    ax = plt.axes()
    ax.axis('equal')
    ax.plot(*room.room.exterior.xy)

    ax.scatter(room.guard_grid[:,0], room.guard_grid[:,1], s = waiting_times, facecolors = 'none', edgecolors = 'r')
    plt.show()


def get_intensities(room, robot_height_scaled, use_strong_visibility = True, use_strong_distances = True):
    # Construct initial intensities matrix, ignoring visibility
    num_guard_points = room.guard_grid.shape[0]
    num_room_points = room.room_grid.shape[0]
    room_intensities = np.zeros(shape = (num_guard_points, num_room_points))

    for guard_idx, guard_pt in enumerate(room.guard_grid):
        for room_idx, room_pt in enumerate(room.room_grid):
            if use_strong_distances:
                room_cell = room.room_cells[room_idx]
                distance_2d = Point(guard_pt).hausdorff_distance(room_cell)
            else:
                distance_2d = norm(guard_pt - room_pt)

            room_intensities[guard_idx, room_idx] = 1/(distance_2d**2 + robot_height_scaled**2)

    # Patch up visibility.
    eps_room = prep(room.room.buffer(EPS))
    broken_sightlines_count = 0
    broken_sightlines_list = []
    for room_idx, room_point in enumerate(room.room_grid):
        if room_idx % 50 == 0: print("Processing room index", room_idx)
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
            print('Unreachable Room Point:', room_idx)
    
    print('Removed', broken_sightlines_count, 'broken sightlines')
    
    return room_intensities


def get_scale(room, waiting_times, scaling_method):
    if scaling_method == 'epsilon':
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
