import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate

# Work-in-progress implementation of https://arxiv.org/pdf/1504.06584.pdf for orthogonal polygons

NUM_DIR = 4
RIGHT = 0
UP    = 1
LEFT  = 2
DOWN  = 3

def construct_orthogonal_polygon(input_polygon, tolerance):
    # Find minimum bounding box and rotate points
    print("Constructing orthogonal polygon")
    bounding_box = input_polygon.minimum_rotated_rectangle
    box_x, box_y = bounding_box.exterior.coords.xy
    delta_x = box_x[1] - box_x[0]
    delta_y = box_y[1] - box_y[0]
    angle = -np.arctan(delta_y/delta_x) #TODO: Is it actually correct to always make this negative?

    rotated_input = rotate(input_polygon, angle, origin = (0,0), use_radians = True)
    x = np.array(rotated_input.exterior.coords.xy[0])
    y = np.array(rotated_input.exterior.coords.xy[1])

    # Solve problem
    orthogonal_polygon, _, _, _, _, _ = simplify_closed_curve(x, y, tolerance)

    # Clean up polygon
    aligned_orthogonal_polygon = rotate(orthogonal_polygon, -angle, origin = (0,0), use_radians = True)
    aligned_orthogonal_polygon = aligned_orthogonal_polygon.buffer(0) # Discard self intersecting portions (TODO: make this more robust)
    aligned_orthogonal_polygon = aligned_orthogonal_polygon.simplify(0) # Get rid of redundant points

    return aligned_orthogonal_polygon



def simplify_closed_curve(x, y, tolerance):
    # First guess - unclosed curve
    segments_needed, integral_error, previous_points, x, y, nearby_points_x, nearby_points_y = simplify_to_orthogonal(x, y, tolerance)

    # Reconstruct polygon
    neighbors = []
    directions = []
    ortho_points_x = []
    ortho_points_y = []
    nearby_idx = 0 # Random guess - assume that this doesn't effect anything on the other side of the polygon (TODO: select the actual best)
    direction = 0 # Random guess (TODO: fix) #TODO: forcing direction is currently broken. Why?
    neighbors.append(nearby_idx)
    ortho_points_x.append(nearby_points_x[-1][nearby_idx])
    ortho_points_y.append(nearby_points_x[-1][nearby_idx])
    for i in np.arange(len(x) - 1, 0, -1):
        nearby_idx, direction = previous_points[i, nearby_idx, direction]
        neighbors.append(nearby_idx)
        directions.append(direction)
        ortho_points_x.append(nearby_points_x[i - 1][nearby_idx])
        ortho_points_y.append(nearby_points_y[i - 1][nearby_idx])

    # Get known point and run again for closed polygon
    known_idx = int(len(x) // 2)
    known_neighbor_idx = neighbors[known_idx]
    known_direction = directions[known_idx]

    rolled_x = np.roll(x[:-1], -known_idx)
    rolled_x = np.append(rolled_x, rolled_x[0])
    rolled_y = np.roll(y[:-1], -known_idx)
    rolled_y = np.append(rolled_y, rolled_y[0])
    segments_needed, integral_error, previous_points, x, y, nearby_points_x, nearby_points_y = simplify_to_orthogonal(rolled_x, rolled_y, tolerance, start_neighbor = known_neighbor_idx, start_dir = known_direction)

    # Reconstruct polygon
    ortho_neighbors = []
    ortho_points_x = []
    ortho_points_y = []
    nearby_idx = known_neighbor_idx
    direction = known_direction
    
    ortho_neighbors.append(nearby_idx)
    ortho_points_x.append(nearby_points_x[-1][nearby_idx])
    ortho_points_y.append(nearby_points_y[-1][nearby_idx])

    for i in np.arange(len(x) - 1, 0, -1):
        nearby_idx, direction = previous_points[i, nearby_idx, direction]
        ortho_neighbors.append(nearby_idx)
        ortho_points_x.append(nearby_points_x[i - 1][nearby_idx])
        ortho_points_y.append(nearby_points_y[i - 1][nearby_idx])

    ortho_polygon = Polygon(zip(ortho_points_x, ortho_points_y))
    return ortho_polygon, segments_needed, integral_error, previous_points, nearby_points_x, nearby_points_y


def simplify_to_orthogonal(x, y, tolerance, start_neighbor = None, start_dir = None):
    FRAC_OF_TOLERANCE = 0.2
    GRID_LINES = tolerance * FRAC_OF_TOLERANCE
    NUM_EACH_SIDE = tolerance // (2 * GRID_LINES)

    num_vertices = len(x)
    num_in_neighborhood = int((2 * NUM_EACH_SIDE + 1) ** 2)
    nearby_points_x = [get_nearby(curr_x, curr_y, GRID_LINES, NUM_EACH_SIDE)[0] for curr_x, curr_y in zip(x,y)]
    nearby_points_y = [get_nearby(curr_x, curr_y, GRID_LINES, NUM_EACH_SIDE)[1] for curr_x, curr_y in zip(x,y)]

    segments_needed = np.full((num_vertices, num_in_neighborhood, NUM_DIR), np.inf)
    integral_error = np.full((num_vertices, num_in_neighborhood, NUM_DIR), np.inf)
    previous_points = np.empty((num_vertices, num_in_neighborhood, NUM_DIR), dtype = object)

    # Base case:
    if start_neighbor is None:
        segments_needed[0,:,:] = 0
        integral_error[0,:,:] = 0
    else:
        if start_dir is None:
            segments_needed[0, start_neighbor, :] = 0
            integral_error[0, start_neighbor , :] = 0
        else:
            segments_needed[0, start_neighbor, :] = 0
            integral_error[0, start_neighbor , :] = 0

    # Initialize arrays for use later (efficiency concerns)
    candidate_segments = np.full(num_in_neighborhood, np.inf)
    candidate_errors = np.full(num_in_neighborhood, np.inf)
    candidate_dirs = np.full(num_in_neighborhood, 0)
    candidate_segments_previous = np.full(NUM_DIR, np.inf)
    candidate_errors_previous = np.full(num_in_neighborhood, np.inf)

    # Induction
    end_idx = 1
    while end_idx < num_vertices:
        end_nearby_x = nearby_points_x[end_idx]
        end_nearby_y = nearby_points_y[end_idx]
        previous_nearby_x = nearby_points_x[end_idx - 1]
        previous_nearby_y = nearby_points_y[end_idx - 1]

        for end_nearby_idx in range(num_in_neighborhood):
            curr_end_point_x = end_nearby_x[end_nearby_idx]
            curr_end_point_y = end_nearby_y[end_nearby_idx]

            for end_dir in range(NUM_DIR):
                candidate_segments.fill(np.inf)
                candidate_errors.fill(np.inf)
                candidate_dirs.fill(0)

                for previous_nearby_idx in range(num_in_neighborhood):
                    previous_nearby_point_x = previous_nearby_x[previous_nearby_idx]
                    previous_nearby_point_y = previous_nearby_y[previous_nearby_idx]
                    
                    candidate_segments_previous.fill(np.inf)
                    candidate_errors_previous.fill(np.inf)
                    
                    previous_directions_allowed = []
                    if previous_nearby_point_x == curr_end_point_x:
                        previous_directions_allowed.append(UP if previous_nearby_point_y < curr_end_point_y else DOWN)
                        approximation_line = curr_end_point_x
                    if previous_nearby_point_y == curr_end_point_y:
                        previous_directions_allowed.append(RIGHT if previous_nearby_point_x < curr_end_point_x else LEFT)
                        approximation_line = curr_end_point_y
                    
                     # Don't allow going backwards
                    opposite_dir = (end_dir + NUM_DIR/2) % NUM_DIR
                    if opposite_dir in previous_directions_allowed:
                        previous_directions_allowed.remove(opposite_dir)

                    # Don't allow going backwards when wrapping around to start
                    # if end_idx == num_vertices - 1 and start_dir is not None:
                    #     opposite_start_dir = (start_dir + NUM_DIR/2) % NUM_DIR
                    #     if opposite_start_dir in previous_directions_allowed:
                    #         previous_directions_allowed.remove(opposite_start_dir)                            

                    # If the previous and current point are the same, force the same direction
                    if (curr_end_point_x == previous_nearby_point_x and curr_end_point_y == previous_nearby_point_y):
                        previous_directions_allowed = [end_dir]

                    # Calculate candidates
                    for previous_dir in previous_directions_allowed:
                        seg_added = 0 if previous_dir == end_dir else 1
                        err_added = get_error(x[end_idx - 1], y[end_idx - 1], x[end_idx], y[end_idx], previous_dir, approximation_line)
                        candidate_segments_previous[previous_dir] = segments_needed[end_idx - 1, previous_nearby_idx, previous_dir] + seg_added
                        candidate_errors_previous[previous_dir] = integral_error[end_idx - 1, previous_nearby_idx, previous_dir] + err_added
                
                    # Select best "previous direction" based on (1) fewest segments, (2) smallest error
                    min_segments = min(candidate_segments_previous)
                    min_indices = np.where(candidate_segments_previous == min_segments)[0]
                    min_error_idx = np.argmin(candidate_errors_previous[min_indices])
                    best_idx = min_indices[min_error_idx] # The best previous direction

                    candidate_segments[previous_nearby_idx] = candidate_segments_previous[best_idx]
                    candidate_errors[previous_nearby_idx] = candidate_errors_previous[best_idx]
                    candidate_dirs[previous_nearby_idx] = best_idx

                # Select best "previous neighbor point" based on (1) fewest segments, (2) smallest error
                min_segments = min(candidate_segments)
                min_indices = np.where(candidate_segments == min_segments)[0]
                min_error_idx = np.argmin(candidate_errors[min_indices])
                best_idx = min_indices[min_error_idx] # The best previous neighbor

                segments_needed[end_idx, end_nearby_idx, end_dir] = candidate_segments[best_idx]
                integral_error[end_idx, end_nearby_idx, end_dir] = candidate_errors[best_idx]
                previous_points[end_idx, end_nearby_idx, end_dir] = (best_idx, candidate_dirs[best_idx])

        if (np.min(segments_needed[end_idx]) == np.inf):
            # Insert a point between previous and new
            to_insert_x = (x[end_idx - 1] + x[end_idx])/2
            to_insert_y = (y[end_idx - 1] + y[end_idx])/2

            to_insert_nearby_x = get_nearby(to_insert_x, to_insert_y, GRID_LINES, NUM_EACH_SIDE)[0]
            to_insert_nearby_y = get_nearby(to_insert_x, to_insert_y, GRID_LINES, NUM_EACH_SIDE)[1]
            
            x = np.insert(x, end_idx, to_insert_x)
            y = np.insert(y, end_idx, to_insert_y)
            num_vertices += 1

            nearby_points_x.insert(end_idx, to_insert_nearby_x)
            nearby_points_y.insert(end_idx, to_insert_nearby_y)

            segments_needed = np.insert(segments_needed, end_idx, np.full(segments_needed[0].shape, np.inf), axis = 0)
            integral_error = np.insert(integral_error, end_idx, np.full(integral_error[0].shape, np.inf), axis = 0)
            previous_points = np.insert(previous_points, end_idx, np.empty(previous_points[0].shape), axis = 0)

        else:
            end_idx += 1

    return (segments_needed, integral_error, previous_points, x, y, nearby_points_x, nearby_points_y)



#####################################
######     Helper Methods      ######
#####################################

def get_nearby(x, y, grid_lines, num_each_side):
    center_x = x // grid_lines * grid_lines
    center_y = y // grid_lines * grid_lines
    start_x = center_x - (num_each_side * grid_lines)
    start_y = center_y - (num_each_side * grid_lines)
    end_x = center_x + ((num_each_side + 1) * grid_lines)
    end_y = center_y + ((num_each_side + 1) * grid_lines)

    nearby_x_ticks = np.arange(start_x, end_x, grid_lines)
    nearby_y_ticks = np.arange(start_y, end_y, grid_lines)
    nearby_x, nearby_y = np.meshgrid(nearby_x_ticks, nearby_y_ticks)
    nearby_x = np.ravel(nearby_x)
    nearby_y = np.ravel(nearby_y)

    return(nearby_x, nearby_y)

def get_error(start_x, start_y, end_x, end_y, direction, approximation):
    if direction == LEFT or direction == RIGHT:
        return max((start_y - approximation)**2, (end_y - approximation)**2)
    else:
        return max((start_x - approximation)**2, (end_x - approximation)**2)

# Input: 1D arrays
# Output: best index, # segments, # error
# Not used right now
def select_best(segments, error):
    min_segments = min(segments)
    candidate_indices = np.where(segments == min_segments)[0]
    best_candidate_idx = np.argmin(error[candidate_indices])
    best_idx = candidate_indices[best_candidate_idx]

    return (best_idx, min_segments, error[best_idx])
