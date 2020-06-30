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

INPUT_FILE = '../floor_plans/gmapping_tenbenham_cropped.png'
OUTPUT_FILE = '../output/tenbenham_polygon.png'
OUTPUT_CSV = '../output/tennbenham_waiting_times.csv'
SOLUTION_FILE = '../output/solution_1.png'
METERS_PER_PIXEL = 0.05
ROBOT_HEIGHT = 1.2192 # in meters (equal to 4 feet)
EPSILON = 0.5
MIN_INTENSITY = 1
THETA = 0

use_strong_visibility = True
use_strong_distances = True
scaling_method = 'none' # must be in {'epsilon', 'branch_and_bound', 'none'}

units_per_pixel =  METERS_PER_PIXEL * 1/ROBOT_HEIGHT
robot_height_pixels = ROBOT_HEIGHT * 1/METERS_PER_PIXEL
print('Scaled robot height', robot_height_pixels)

print('Extracting polygon')
polygon = extract_polygon(INPUT_FILE, OUTPUT_FILE)
print('Creating room')
room = Room(polygon, units_per_pixel, room_eps = EPSILON, guard_eps = EPSILON, guard_scale = 1)
print('Solving lp')
time, waiting_times, intensities = solve_full_lp(room, robot_height_pixels, use_strong_visibility, use_strong_distances, scaling_method, MIN_INTENSITY)
print('Solution Time:', time)
print(room.guard_grid.shape)
print(intensities.shape)

rows = []
for (x, y), t in zip(room.guard_grid, waiting_times):
    if t > 1e-3:
        rows.append({'x': x, 'y': y, 'theta': THETA, 'time': t})
pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

print('Visualizing solution')
visualize_times(room, waiting_times, SOLUTION_FILE)
