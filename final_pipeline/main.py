# Input: filepath to png
#        Constants (how? separate file? args? eg. method)
#        Output filepath

# Pipeline:
#  1. Room segmentation: get polygon (geojson? probably save two files, geojson and png)
#  2. Pass polygon to Room constructor
#  3. Pass Room to lp solver
#  4. Get output images and txt file from solution
import pandas as pd
from room import Room
from polygon_extraction import extract_polygon
from lp_solver import solve_full_lp, visualize_times

# I/O Files
INPUT_FILE = '../floor_plans/gmapping_tenbenham_cropped.png'
OUTPUT_FILE = '../output/tenbenham_polygon.png'
OUTPUT_CSV = '../output/tennbenham_waiting_times.csv'
SOLUTION_FILE = '../output/solution_1.png'

# Robot dimensions
METERS_PER_PIXEL = 0.05 # Constant to determine the correspondence between input pixels and dimensions in real space
                        # HARDCODED. Should read from input file instead
ROBOT_HEIGHT = 1.2192   # in meters (currently set to 4 feet)
EPSILON = 0.5           # Tolerance error. XXX What is the unit?
MIN_INTENSITY = 1       # XXX what is this? What is the unit?
THETA = 0               # XXX what is this? What is the unit?

# Algorithm parameters. See documentation for the different variations
use_strong_visibility = True
use_strong_distances = True
scaling_method = 'none' # must be in {'epsilon', 'branch_and_bound', 'none'}

units_per_pixel =  METERS_PER_PIXEL * 1/ROBOT_HEIGHT
robot_height_pixels = ROBOT_HEIGHT * 1/METERS_PER_PIXEL
print('Scaled robot height', robot_height_pixels)

# Step 1: read input file (pixel-like image) and transform it to a simple polygon (with clearly marked in/out)
print('Extracting polygon')
polygon = extract_polygon(INPUT_FILE, OUTPUT_FILE)

# Step 2: a Room object contains not only the boundary, but creates a discretized list of places
#         for the robot to guard (and list of places where the robot can actually move to)
print('Creating room')
room = Room(polygon, units_per_pixel, room_eps = EPSILON, guard_eps = EPSILON, guard_scale = 1)

# Step 3: we generate the LP problem and solve it.
print('Solving lp')
time, waiting_times, intensities = solve_full_lp(room, robot_height_pixels, use_strong_visibility, use_strong_distances, scaling_method, MIN_INTENSITY)

# Step 4: Output a solution
print('Total solution time:', time)
print(room.guard_grid.shape)
print(intensities.shape)

# Create a csv of all positions and waiting time
rows = []
for (x, y), t in zip(room.guard_grid, waiting_times):
    # We drop points that you stop less than a milisecond. HARDCODED
    if t > 1e-3:
        rows.append({'x': x, 'y': y, 'theta': THETA, 'time': t})
pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

# Graphical visualizations of the solution
print('Visualizing solution')
visualize_times(room, waiting_times, SOLUTION_FILE)
