import cvxpy as cp
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import box, Point, LineString, Polygon, MultiPolygon
import shapely.geometry
from shapely.affinity import scale
from shapely.prepared import prep
from shapely.ops import nearest_points
from scipy.spatial import distance_matrix
from matplotlib.animation import FuncAnimation
from numpy.linalg import norm
from skgeom import *
from sys import exit

from branch_and_bound import branch_bound_poly

EPS = 1e-5 # Arbitrary small number to avoid rounding error

def solve_full_lp(room, robot_height_scaled, robot_radius_scaled, use_strong_visibility, use_strong_distances, scaling_method, min_intensity):
    room_intensities = get_intensities(room, robot_height_scaled, robot_radius_scaled, use_strong_visibility, use_strong_distances)
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

def visualize_times(room, waiting_times, plot_point = None):
    ax = plt.axes()
    ax.axis('equal')
    ax.plot(*room.room.exterior.xy)
    ax.plot(*room.guard.exterior.xy)

    ax.scatter(room.guard_grid[:,0], room.guard_grid[:,1], s = 5, facecolors = 'r', alpha = 0.5)
    ax.scatter(room.room_grid[:,0], room.room_grid[:,1], s = 5, facecolors = 'black', alpha = 0.5)

    ax.scatter(room.guard_grid[:,0], room.guard_grid[:,1], s = waiting_times, facecolors = 'none', edgecolors = 'r', alpha = 0.5)

    for cell in room.room_cells:
        ax.plot(*cell.exterior.xy, color = 'black', alpha = 0.5)

    if plot_point is not None:
        ax.scatter(plot_point[0], plot_point[1], s = 10, facecolors = 'orange')

    plt.show()


def get_intensities(room, robot_height_scaled, robot_radius_scaled, use_strong_visibility = True, use_strong_distances = True):
    # 1. Make sure that every room region can be seen from somewhere in the guard grid
    eps_room = room.room.buffer(EPS)
    eps_room_prepared = prep(eps_room)
    guard_region = room.room.buffer(-robot_radius_scaled)

    total_added = 0
    for room_idx, room_pt in enumerate(room.room_grid):
        none_visible = True
        for guard_idx, guard_point in enumerate(room.guard_grid):
            if use_strong_visibility:
                # If a point can see all vertices of a simple polygon,
                # then it can see the entire polygon
                room_cell = room.room_cells[room_idx]
                room_cell_points = list(room_cell.exterior.coords) 
                sightlines = [LineString([pt, guard_point]) for pt in room_cell_points]
                is_visible = all([eps_room_prepared.contains(line) for line in sightlines])
                is_visible = is_visible & (Point(guard_point).distance(room_cell) >= robot_radius_scaled)
            else:
                sight = LineString([guard_point, room_point])
                is_visible = eps_room_prepared.contains(sight)
            if is_visible:
                none_visible = False
                break
        if none_visible:
            print("Adding point that can see room point", room_pt)
            if use_strong_visibility:
                room_cell = room.room_cells[room_idx]
                room_cell_points = list(room_cell.exterior.coords)
                visibility_polygons = [compute_visibility_polygon(eps_room, Point(point)) for point in room_cell_points]
                # Check 1 pixel farther away than necessary to avoid approximation errors with buffer(). TODO make this more robust
                shadow_polygons = [Point(point).buffer(robot_radius_scaled + 1) for point in room_cell_points] 
                result = visibility_polygons[0]
                for i in range(1, len(visibility_polygons)):
                    result = result.intersection(visibility_polygons[i])
                for i in range(0, len(shadow_polygons)):
                    result = result.difference(shadow_polygons[i])
                result = result.intersection(guard_region)
                
                try:
                    point_to_add_shapely, _ = nearest_points(result, Point(room_pt))
                    point_to_add = np.array([point_to_add_shapely.x, point_to_add_shapely.y])
                except Exception as e:
                    if str(e) == 'The first input geometry is empty':
                        print('ERROR: Problem is infeasible.')
                        print('       No point in the guard region can see', room_pt)
                        exit()
                    else:
                        raise e
                
                

                print("Adding point", point_to_add)
                room.guard_grid = np.append(room.guard_grid, [point_to_add], axis = 0)
                room.guard_cells = np.append(room.guard_cells, None) # TODO: Check that room.guard_cells is not used anywhere...
                total_added += 1
            else:
                # TODO: adapt this to work with non-strong visibility
                ...

    print("Added", total_added, "points to make problem feasible")
    
    # 2. Construct initial intensities matrix, ignoring visibility
    num_guard_points = room.guard_grid.shape[0]
    num_room_points = room.room_grid.shape[0]
    room_intensities = np.zeros(shape = (num_guard_points, num_room_points))

    for guard_idx, guard_pt in enumerate(room.guard_grid):
        if guard_idx % 50 == 0: print("Getting intensity for point: ", guard_idx)
        for room_idx, room_pt in enumerate(room.room_grid):
            if use_strong_distances:
                room_cell = room.room_cells[room_idx]
                distance_2d = Point(guard_pt).hausdorff_distance(room_cell)
                dist_for_shadow = Point(guard_pt).distance(room_cell) # Shortest distance: worst case scenario
            else:
                distance_2d = norm(guard_pt - room_pt)
                dist_for_shadow = distance2d
            angle = np.pi/2 - np.arctan(robot_height_scaled/distance_2d)

            # Account for robot shadow
            if dist_for_shadow < robot_radius_scaled:
                room_intensities[guard_idx, room_idx] = 0
            else:
                # Energy received follows an inverse-square law and Lambert's cosine law
                room_intensities[guard_idx, room_idx] = np.cos(angle) * 1/(distance_2d**2 + robot_height_scaled**2)

    # 3. Adjust for visibility
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
                is_visible = is_visible & (Point(guard_point).distance(room_cell) >= robot_radius_scaled)
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
            print('Unreachable Room Point:', room_point)
    
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

#############################
## Code for naive solution ##
#############################

# Naive solution:
#   Place a point near the center of the room and illuminate all parts of the room it can see
#   Repeat the process for the regions that have not yet been covered
def solve_naive(room, robot_height_pixels, min_intensity):
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
        curr_time = min_intensity * (max_distance**2 + robot_height_pixels**2)
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
# 'point' cannot be on the boundary of the polygon
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
