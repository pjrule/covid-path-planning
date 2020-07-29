# Main script to solve the UV Light optimization problem

import pandas as pd
from room import Room
from polygon_extraction import extract_polygon
from lp_solver import solve_full_lp, visualize_times, solve_naive
from shapely.geometry import box

######################
###   Parameters   ###
######################

# I/O Files
INPUT_FILE = '../floor_plans/gmapping_tenbenham_cropped.png' # Alternative: '../floor_plans/hrilab_2510_sled_cropped.png'
OUTPUT_FILE = '../output/tenbenham_polygon.png'
OUTPUT_CSV = '../output/tenbenham_waiting_times.csv'

# Robot dimensions
METERS_PER_PIXEL = 0.05 # Constant to determine the correspondence between input pixels and dimensions in real space
                        # HARDCODED. Should read from input file instead
ROBOT_HEIGHT = 1.2192   # in meters (currently set to 4 feet)
ROBOT_RADIUS = 0.48     # "Radius" of the robot, in meters
                        #   The center of the robot will stay at least ROBOT_RADIUS meters from walls
                        #   The robot will cast a circular shadow with radius ROBOT_RADIUS meters
CONTOUR_ACCURACY = 5   # Acceptable error when simplifying input polygon
                        #   A smaller value simplifies less. Smaller values
                        #   preserve room details but also preserves noise.
                        #   The best value should be determined experimentally
GUARD_EPSILON = 0.5     # Controls the fineness of the grid of possible
                        #   locations the robot is permitted to irradiate from,
                        #   in units of ROBOT_HEIGHT meters.
                        #   A smaller epsilon guarantees that we find a
                        #   solution closer to optimal, assuming infinite speed
                        #   The right value should be determined experimentally
ROOM_EPSILON = 0.5      # Controls the fineness of the grid of room regions.
                        #   A smaller epsilon guarantees that we find a
                        #   better solution
MIN_INTENSITY = 1       # Threshold of energy each point in the room must
                        #   receive, assuming that the UV light emits
                        #   1/distance_in_pixels^2 units of energy

# Algorithm parameters. See documentation for the different variations
naive_solution = False
use_strong_visibility = True
use_strong_distances = True
scaling_method = 'none' # must be in {'epsilon', 'branch_and_bound', 'none'}


############################
###   Compute Solution   ###
############################

units_per_pixel =  METERS_PER_PIXEL * 1/ROBOT_HEIGHT
robot_height_pixels = ROBOT_HEIGHT * 1/METERS_PER_PIXEL
robot_radius_pixels = ROBOT_RADIUS * 1/METERS_PER_PIXEL
print('Scaled robot height', robot_height_pixels)

# Step 1: read input file (pixel-like image) and transform it to a simple polygon (with clearly marked in/out)
print('Extracting polygon')
polygon = extract_polygon(INPUT_FILE, OUTPUT_FILE, contour_accuracy = CONTOUR_ACCURACY)

# Step 2: a Room object contains not only the boundary, but creates a discretized list of places
#         for the robot to guard (and list of places where the robot can actually move to)
print('Creating room')
room = Room(polygon, units_per_pixel, robot_buffer_pixels = robot_radius_pixels, room_eps = ROOM_EPSILON, guard_eps = GUARD_EPSILON)

if naive_solution:
    solve_naive(room, robot_height_pixels, MIN_INTENSITY)
else:
    # Step 3: we generate the LP problem and solve it.
    print('Solving lp')
    time, waiting_times, intensities = solve_full_lp(room, robot_height_pixels, robot_radius_pixels, use_strong_visibility, use_strong_distances, scaling_method, MIN_INTENSITY)

    # Step 4: Output a solution
    print('Total solution time:', time)
    print(room.guard_grid.shape)
    print(intensities.shape)

    # Create a csv of all positions and waiting time
    rows = []
    for (x, y), t in zip(room.guard_grid, waiting_times):
        # We drop points that you stop less than a milisecond. HARDCODED
        if t > 1e-3:
            rows.append({'x': x, 'y': y, 'time': t})
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    # Graphical visualizations of the solution
    print('Visualizing solution')
    visualize_times(room, waiting_times)
