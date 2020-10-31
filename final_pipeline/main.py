# Main script to solve the UV Light optimization problem

import pandas as pd
from room import Room
from polygon_extraction import extract_polygon, construct_isValidLocation_function
from lp_solver import solve_full_lp, visualize_times, solve_naive, visualize_energy, visualize_distance
from shapely.geometry import box
import matplotlib.pyplot as plt
from shapely.ops import transform
import sys

######################
###   Parameters   ###
######################

# I/O Files
#INPUT_FILE = '../floor_plans/hrilab_2510_sled.pgm'
#INPUT_YAML = '../floor_plans/hrilab_2510_sled.yaml'
INPUT_FILE = '../floor_plans/2560.pgm'
INPUT_YAML = '../floor_plans/2560.yaml'
OUTPUT_CSV = '../output/waiting_times.csv'

# Environment parameters
ROBOT_HEIGHT = 1.2192         # Height of UV light, in meters
ROBOT_RADIUS = 0.4            # Distance from robot center to farthest point,
                              # in meters
ROBOT_WATTAGE = 55            # Power of the UV light in Watts (ie. J/sec)
DISINFECTION_THRESHOLD = 1206 # Joules/meter^2

# Preprocessing parameters
ORTHOGONAL_TOL = 40           # Tolerance for orthogonal simplification, in pixels
AVOID_UNKNOWN_REGIONS = True  # Treat "unknown" pixels as walls when determining
                              #  the spaces that the robot can move to

# Algorithm parameters. See documentation for the different variations
naive_solution = False
use_weak_everything = False # Compute a lower bound on the time for a solution
                            # Overrides use_strong_visibility and use_strong_distances
use_strong_visibility = True
use_strong_distances = False
use_shadow = False
scaling_method = 'branch_and_bound'     # must be in {'epsilon', 'branch_and_bound', 'none'}

ROBOT_EPSILON = 0.2     # Size of grid for discretization of possible robot
                        #   locations, in meters
ROOM_EPSILON = 0.2      # Size of grid for discretization of locations to
                        #   disinfect, in meters
                        # Smaller epsilon values guarantee that we find a
                        #   solution closer to optimal, assuming infinite speed
                        #   The right value should be determined experimentally

show_visualizations = False

############################
###   Compute Solution   ###
############################

# Step 1: read input file (pixel-like image) and transform it to a simple polygon
#         (with clearly marked in/out)
print('Extracting polygon')
polygon_data = extract_polygon(INPUT_FILE,
                               INPUT_YAML,
                               ortho_tolerance = ORTHOGONAL_TOL,
                               show_visualization = show_visualizations)
polygon, gray_img, xy_to_pixel, meters_per_pixel = polygon_data

is_valid_location = construct_isValidLocation_function(gray_img,
                                                       xy_to_pixel,
                                                       ROBOT_RADIUS,
                                                       meters_per_pixel,
                                                       AVOID_UNKNOWN_REGIONS)

# Step 2: a Room object contains not only the boundary, but creates a
#         discretized list of places for the robot to guard (and list
#         of places where the robot can actually move to)
print('Creating room')
room = Room(polygon,
            gray_img,
            xy_to_pixel,
            robot_buffer_meters = ROBOT_RADIUS,
            is_valid_guard = is_valid_location,
            room_eps = ROOM_EPSILON,
            guard_eps = ROBOT_EPSILON,
            show_visualization = show_visualizations)

if naive_solution:
    solve_naive(room, ROBOT_HEIGHT, DISINFECTION_THRESHOLD)
else:
    # Step 3: we generate the LP problem and solve it.
    print('Solving lp')
    lp_solution_data = solve_full_lp(room,
                                     ROBOT_HEIGHT,
                                     ROBOT_RADIUS,
                                     ROBOT_WATTAGE,
                                     use_weak_everything,
                                     use_strong_visibility,
                                     use_strong_distances,
                                     scaling_method,
                                     DISINFECTION_THRESHOLD,
                                     show_visualizations,
                                     use_shadow)
    time, waiting_times, intensities, unguarded_room_idx, _, percent_disinfected = lp_solution_data

    # Step 4: Output a solution
    print("-"*80)
    print('Total solution time:', time)
    print('Percent Disinfected:', percent_disinfected)
    print("-"*80)

    # Create a csv of all positions and waiting time
    rows = []
    for (x, y), t in zip(room.guard_grid, waiting_times):
        # We drop points that you stop less than a milisecond. HARDCODED
        if t > 1e-3:
            rows.append({'x': x, 'y': y, 'time': t})
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    # Graphical visualizations of the solution
    if show_visualizations:
        print('Visualizing solution')
        visualize_times(room, waiting_times, unguarded_room_idx)
        visualize_energy(room, waiting_times, intensities, DISINFECTION_THRESHOLD)
        visualize_distance(room, waiting_times, intensities)
