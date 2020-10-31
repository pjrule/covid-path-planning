import cvxpy as cp
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from shapely.geometry import box, Point, LineString, Polygon, MultiPolygon
import shapely.geometry
from shapely.affinity import scale
from shapely.prepared import prep
from scipy.spatial import distance_matrix
from matplotlib.animation import FuncAnimation
from numpy.linalg import norm
from sys import exit
from shapely.ops import transform, unary_union

# Try to load skgeom package (for naive solution)
SKGEOM_AVAIL = False
try:
    from skgeom import *
    SKGEOM_AVAIL = True
except ImportError as e:
    pass

from branch_and_bound import branch_bound_poly

EPS = 1e-5 # Arbitrary small number to avoid rounding error

def solve_full_lp(room,
                  robot_height,
                  robot_radius,
                  robot_power,
                  use_weak_everything,
                  use_strong_visibility,
                  use_strong_distances,
                  scaling_method,
                  min_intensity,
                  show_visualization,
                  use_shadow):
    intensity_tuple = get_intensities(room,
                                      robot_height,
                                      robot_radius,
                                      robot_power,
                                      use_weak_everything,
                                      use_strong_visibility,
                                      use_strong_distances,
                                      show_visualization,
                                      use_shadow)
    room_intensities, unguarded_room_idx = intensity_tuple

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
    
    disinfected_region, percent_disinfected = calculate_percent_disinfected(
                                                room,
                                                unguarded_room_idx,
                                                loc_times.value,
                                                use_strong_distances,
                                                use_weak_everything)

    unscaled_time = loc_times.value.sum()
    scale = get_scale(room, min_intensity, disinfected_region, loc_times.value,
                      scaling_method, robot_power, robot_height, robot_radius, use_shadow)
    solution_time = scale * unscaled_time

    # TODO: Filter for significant points

    return solution_time, loc_times.value, room_intensities.T, unguarded_room_idx, disinfected_region, percent_disinfected

def calculate_percent_disinfected(room, not_visible_room_idx, waiting_times, use_strong_distances, use_weak_everything):
    # If we use strong/weak visibility, we guarantee disinfection in discrete
    #    grid cells (the epsilon-neighborhoods)
    if use_strong_distances or use_weak_everything:
        visible_cells       = [room.full_room_cells[i]
                               for i in range(room.full_room_cells.shape[0])
                               if i not in not_visible_room_idx]
        disinfected_region  = unary_union(visible_cells)
        
    # Otherwise, assume we use branch-and-bound to scale up solution
    else:
        print("""
               Note: When calculating percent disinfected, we assume that
               a branch-and-bound method will be used to scale the solution
               appropriately.""")
        # HARDCODED: Ignore points where we wait less than a millisecond
        vis_preprocessing       = compute_skgeom_visibility_preprocessing(room.room)
        waiting_points          = [point for i, point in enumerate(room.guard_grid)
                                   if waiting_times[i] > 1e-3]
        all_visibility_polygons = [compute_visibility_polygon(vis_preprocessing,
                                   point) for point in waiting_points]
        disinfected_region      = shapely.ops.unary_union(all_visibility_polygons).simplify(0.001)


    area_disinfected    = disinfected_region.area
    total_area          = room.room.area
    percent_disinfected = (area_disinfected/total_area) * 100


    print("Percent of room that can be disinfected: {:.2f}%".format(
            percent_disinfected))

    return disinfected_region, percent_disinfected

def visualize_times(room, waiting_times, unguarded_room_idx):
    ax = plt.axes()
    ax.axis('equal')

    # Plot room (image and polygon)
    ax.imshow(room.room_img)
    ax.plot(*transform(room.xy_to_pixel, room.room).exterior.xy, 'black')

    # Plot guard points
    guard_points_x = [room.xy_to_pixel(x, y)[0] for x, y in room.guard_grid]
    guard_points_y = [room.xy_to_pixel(x, y)[1] for x, y in room.guard_grid]
    ax.scatter(guard_points_x, guard_points_y, s = waiting_times / 10, facecolors = 'none', edgecolors = 'r')


    # Plot unguarded regions
    unguarded = [room.full_room_cells[i] for i in unguarded_room_idx]
    for idx, cell in zip(unguarded_room_idx, unguarded):
        if room.full_room_iswall[idx] == 0:
            ax.fill(*transform(room.xy_to_pixel, cell).exterior.xy, "darkred", alpha = 0.6)
        elif room.full_room_iswall[idx] == 1:
            ax.fill(*transform(room.xy_to_pixel, cell).xy, "darkred", alpha = 0.6)
        else:
            raise Exception("Unable to classify room cell: ", cell)

    # Plot unenterable regions
    unenterable_indices = []
    for i, room_cell in enumerate(room.full_room_cells):
        is_visible = False
        prep_room_cell = prep(room_cell)
        for guard_pt in room.guard_grid:
            if prep_room_cell.contains(Point(*guard_pt)):
                is_visible = True
                break
        if not is_visible and i not in unguarded_room_idx:
            unenterable.append(i)

    for idx in unenterable_indices:
        if room.full_room_iswall[idx] == 0:
            ax.fill(*transform(room.xy_to_pixel, room.full_room_cells[idx]).exterior.xy, "b", alpha = 0.6)
        elif room.full_room_iswall[idx] == 1:
            ax.fill(*transform(room.xy_to_pixel, room.full_room_cells[idx]).xy, "b", alpha = 0.6)
        else:
            raise Exception("Unable to classify room cell: ", room.full_room_cells[idx])

    plt.show()

def visualize_energy(room, waiting_times, intensities, threshold):
    ax = plt.axes()
    ax.axis('equal')

    # Plot room (image and polygon)
    ax.imshow(room.room_img)
    ax.plot(*transform(room.xy_to_pixel, room.room).exterior.xy, 'black')

    for i, cell in enumerate(room.full_room_cells):
        energy_per_area = intensities[i] @ waiting_times
        if energy_per_area == threshold:
            print("EXACTLY THRESHOLD: ", room.full_room_grid[i])
            color = cm.YlGnBu(1)
        elif energy_per_area <= threshold * 1.2:
            color = cm.YlGnBu(0.75)
        elif energy_per_area <= threshold * 1.5:
            color = cm.YlGnBu(0.5)
        else:
            color = cm.YlGnBu(0.25)
        
        if room.full_room_iswall[i] == 0:
            ax.fill(*transform(room.xy_to_pixel, cell).exterior.xy, color = color)
        elif room.full_room_iswall[i] == 1:
            ax.fill(*transform(room.xy_to_pixel, cell).xy, color = color)
        else:
            raise Exception("Unable to classify room cell: ", cell)

    plt.show()


def visualize_distance(room, waiting_times, intensities):
    ax = plt.axes()
    ax.axis('equal')

    # Plot room (image and polygon)
    ax.imshow(room.room_img)
    ax.plot(*transform(room.xy_to_pixel, room.room).exterior.xy, 'black')


    # Upper bound on maximum distance between any two points
    # Used as a heuristic for scaling
    max_dist = room.room.centroid.hausdorff_distance(room.room)

    for i, cell in enumerate(room.full_room_cells):
        disinfection_per_robot_location = np.multiply(intensities[i], waiting_times)
        distance_per_robot_location = [norm(room.full_room_grid[i] - robot_loc) for robot_loc in room.guard_grid]
        avg_dist = np.average(distance_per_robot_location, weights = disinfection_per_robot_location)
        color = cm.YlGnBu(avg_dist/max_dist)
        
        if room.full_room_iswall[i] == 0:
            ax.fill(*transform(room.xy_to_pixel, cell).exterior.xy, color = color)
        elif room.full_room_iswall[i] == 1:
            ax.fill(*transform(room.xy_to_pixel, cell).xy, color = color)
        else:
            raise Exception("Unable to classify room cell: ", cell)

    plt.show()


# Intensity is power transmitted per unit area
def get_intensities(room, robot_height, robot_radius, robot_power, use_weak_everything = False, use_strong_visibility = True, use_strong_distances = True, show_visualization = True, use_shadow = True):
    # Construct initial intensities matrix, ignoring visibility
    # *Do* account for inverse distance, angle, and robot shadow
    num_guard_points = room.guard_grid.shape[0]
    num_room_points = room.full_room_grid.shape[0]
    room_intensities = np.zeros(shape = (num_guard_points, num_room_points))

    for guard_idx, guard_pt in enumerate(room.guard_grid):
        if guard_idx % 50 == 0: print("Getting intensity for guard point: {}/{}".format(guard_idx, room.guard_grid.shape[0]))
        for room_idx, room_pt in enumerate(room.full_room_grid):
            if use_weak_everything:
                guard_cell = room.guard_cells[guard_idx]
                distance_2d = Point(room_pt).distance(guard_cell)
                dist_for_shadow = Point(room_pt).hausdorff_distance(guard_cell)
            elif use_strong_distances:
                room_cell = room.full_room_cells[room_idx]
                distance_2d = Point(guard_pt).hausdorff_distance(room_cell)
                dist_for_shadow = Point(guard_pt).distance(room_cell) # Shortest distance: worst case scenario
            else:
                distance_2d = norm(guard_pt - room_pt)
                dist_for_shadow = distance_2d

            if room.full_room_iswall[room_idx]:
                angle = np.pi/2 if distance_2d == 0 else np.arctan(robot_height/distance_2d)
            else:
                angle = 0 if distance_2d == 0 else np.pi/2 - np.arctan(robot_height/distance_2d)

            # Account for robot shadow
            if use_shadow and dist_for_shadow < robot_radius:
                room_intensities[guard_idx, room_idx] = 0
            else:
                # Energy received follows an inverse-square law and Lambert's cosine law
                room_intensities[guard_idx, room_idx] = robot_power * np.cos(angle) * 1/(4 * np.pi * (distance_2d**2 + robot_height**2))

    # Patch up visibility.
    eps_room = prep(room.room.buffer(EPS))
    room_vis_preprocessing = compute_skgeom_visibility_preprocessing(room.room.buffer(EPS))

    # Precompute visibility polygons
    print("Precomputing visibility polygons")
    if use_weak_everything:
        room_visibility = [None]*room.full_room_grid.shape[0]
        for i, room_pt in enumerate(room.full_room_grid):
            room_visibility[i] = compute_visibility_polygon(room_vis_preprocessing, room_pt)
    elif use_strong_visibility:
        guard_visibility = [None]*room.guard_grid.shape[0]
        for i, guard_pt in enumerate(room.guard_grid):
            guard_visibility[i] = compute_visibility_polygon(room_vis_preprocessing, guard_pt)
    print("Finished Precomputing visibility polygons")

    broken_sightlines_count = 0
    broken_sightlines_list = []
    not_visible_room_idx = []
    for guard_idx, guard_point in enumerate(room.guard_grid):
        if guard_idx % 50 == 0: print("Processing guard index: {}/{}".format(guard_idx, room.guard_grid.shape[0]))
        for room_idx, room_point in enumerate(room.full_room_grid):
            if use_weak_everything:
                # Check that *room* can see at least one point in guard cell
                room_point_vis = room_visibility[room_idx]
                guard_cell = room.guard_cells[guard_idx]
                is_visible = not room_point_vis.intersection(guard_cell).is_empty
                
            elif use_strong_visibility:
                # If a point can see all vertices of a simple polygon,
                # then it can see the entire polygon
                room_cell = room.full_room_cells[room_idx]

                if room.full_room_iswall[room_idx] == 0: # Floor cell
                    room_cell_points = list(room_cell.exterior.coords)
                elif room.full_room_iswall[room_idx] == 1: # Wall cell
                    room_cell_points = list(room_cell.coords)
                else:
                    raise Exception("Unable to classify room cell: ", room_cell)

                sightlines = [LineString([pt, guard_point]) for pt in room_cell_points]
                is_visible = all([eps_room.contains(line) for line in sightlines])
            else:
                sight = LineString([guard_point, room_point])
                is_visible = eps_room.contains(sight)
            if not is_visible:
                broken_sightlines_list.append((guard_point, room_point))
                broken_sightlines_count += 1
                room_intensities[guard_idx, room_idx] = 0

    # Ignore unreachable room points (do not guarantee that they are illuminated)
    # All intensities are lower than 'robot_power', so we must spend at least
    # 'threshold'/'robot_power' time in the solution.
    # By assigning all unreachable room points a fake intensity of 'robot_power'
    # everywhere, we guarantee that any solution that covers the reachable
    # room points will also "cover" the unreachable room points
    # In other words, the algorithm will ignore the unreachable room points
    not_visible_room_idx = np.argwhere(np.max(room_intensities, axis = 0) == 0).flatten()
    room_intensities[:, not_visible_room_idx] = robot_power
    if len(not_visible_room_idx) > 0:
        print('CAUTION: Not all points in the room can be illuminated')
        print('         Red regions will not be disinfected by the robot')
        print('(Visualization assume "strong distance" algorithm is used:')
        print(' the "branch-and-bound" algorithm will disinfect slightly more.)')
        if show_visualization:
            plt.imshow(room.room_img)
            plt.plot(*transform(room.xy_to_pixel, room.room).exterior.coords.xy, 'black')

            for i in not_visible_room_idx:
                if room.full_room_iswall[i] == 0:
                    plt.fill(*transform(room.xy_to_pixel, room.full_room_cells[i]).exterior.coords.xy, 'red')
                elif room.full_room_iswall[i] == 1:
                    plt.fill(*transform(room.xy_to_pixel, room.full_room_cells[i]).coords.xy, 'red')
                else:
                    raise Exception("Unable to classify room cell: ", room.full_room_cells[i])
            plt.show()

    
    print('Removed', broken_sightlines_count, 'broken sightlines')
    return room_intensities, not_visible_room_idx


# Note: disinfection_region is a shapely object (I *think* either a Polygon
#       or Multipolygon) representing the region that we guarantee receives
#       at least some illumination
def get_scale(room, min_intensity, disinfected_region, waiting_times, scaling_method,
              robot_power, robot_height, robot_radius, use_shadow):
    if scaling_method == 'epsilon':
        print('ERROR: Epsilon scaling not currently supported. Need to convert epsilon to units of robot height')
        return None
        scale = (np.sqrt(room.room_eps**2 + 4) + room.room_eps)/(np.sqrt(room.room_eps**2 + 4) - room.room_eps)
    elif scaling_method == 'branch_and_bound':
        full_room = room.room
        room.room = disinfected_region
        _, lower_bound, _ = branch_bound_poly(
            room,
            waiting_times,
            robot_height=robot_height,
            robot_radius=robot_radius,
            shadow=use_shadow,
            max_iters=1000
        )
        room.room = full_room
        scale = min_intensity/(robot_power*lower_bound)
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
def solve_naive(room, robot_height, robot_radius, min_intensity, use_shadow):
    NUM_SOLUTION_POINTS = 10

    if not SKGEOM_AVAIL:
        exit('Error: Naive solution is not available (skgeom library could not be loaded)')

    plt.close()
    covered_regions = None # Multipolygon
    solution_points = []
    solution_times = []

    not_covered = room.room
    time = 0


    # Note: The scikit-geometry code will fail if a candidate solution point is
    #       every a vertex. This should never happen because we select solution
    #       points from the guard region
    vis_preprocessing = compute_skgeom_visibility_preprocessing(room.room)
    candidate_points = random_points_in_polygon(room.guard, NUM_SOLUTION_POINTS)
    not_covered = room.room

    for point in candidate_points:
        if room.guard.intersection(not_covered).is_empty: # Cannot add any more points
            break

        vis = compute_visibility_polygon(vis_preprocessing, (point.x, point.y))
        to_be_covered = not_covered.intersection(vis)
        if use_shadow:
            to_be_covered = to_be_covered.difference(point.buffer(robot_radius))

        if to_be_covered.is_empty:
            continue
        else:
            not_covered = not_covered.difference(vis)
            max_distance = point.hausdorff_distance(to_be_covered)
            curr_time = min_intensity * (max_distance**2 + robot_height**2)
            time += curr_time
            solution_points.append(point)
            solution_times.append(curr_time)
        #plot_multi(room.room, to_be_covered, solution_points, 'orange')
        #plot_multi(room.room, not_covered, solution_points)
    
    percent_disinfected = (1 - not_covered.area/room.room.area) * 100
    print('Total solution time:', time)
    print('Area disinfected:', percent_disinfected)

    #fig, ax = plt.subplots()
    #ax.set_aspect('equal')
    #ax.plot(*room.room.exterior.xy)
    #ax.scatter([pt.x for pt in solution_points], [pt.y for pt in solution_points], s = np.array(solution_times)/10, facecolors = 'none', edgecolors = 'r')
    #plt.show()

    # Return a tuple in the form
    #  (time, waiting_times, intensities, unguarded_room_idx, percent_disinfected)
    return time, solution_times, percent_disinfected

# Returns an array of 'num_points' random points contained by the shapely
#   polygon
def random_points_in_polygon(polygon, num_points):
    prepared_polygon = prep(polygon)
    points = []
    minx, miny, maxx, maxy = polygon.bounds

    while len(points) < num_points:
        point = Point(np.random.uniform(minx, maxx),
                      np.random.uniform(miny, maxy))
        if prepared_polygon.contains(point):
            points.append(point)
    
    return points

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


# Returns the scikit-geometry RotationalSweepVisibility object and arrangement obejct
#   associated with the input shapely polygon
def compute_skgeom_visibility_preprocessing(polygon):
    points_list = list(polygon.exterior.coords) # Includes a repeating point at the end
    xy_segments = [(points_list[i], points_list[i+1]) for i in range(len(points_list) - 1)]
    segments = [Segment2(Point2(x1, y1), Point2(x2, y2)) for ((x1, y1), (x2, y2)) in xy_segments]

    arr = arrangement.Arrangement()
    for s in segments:
        arr.insert(s)

    vs = RotationalSweepVisibility(arr)

    return (vs, arr)

# Find the visibility polygon of a point
# Input: a scikit-geometry RotationalSweepVisibility object describing the room
#        a point (x,y)
# Output: shapely polygon describing the visibility polygon of the point
def compute_visibility_polygon(skgeom_vis_preprocessing, point):
    vs, arr = skgeom_vis_preprocessing
    pt = Point2(point[0], point[1])
    face = arr.find(pt)
    vx = vs.compute_visibility(pt, face)

    visibility_polygon_points = [(v.curve().point(0).x(), v.curve().point(0).y()) for v in vx.halfedges]
    return(shapely.geometry.Polygon(visibility_polygon_points))

# Find the visibility polygon of a point
# Inputs and output are shapely objects
# Converts to scikit-geometry to find visibility intersection
# Note: this is inefficient for repeated calls on the same shapely polygon
def compute_visibility_polygon_from_shapely(polygon, point):
    skgeom_vis_preprocessing = compute_rotational_sweep(polygon)
    return(compute_visibility_polygon(skgeom_vis_preprocessing, (point.x, point.y)))
