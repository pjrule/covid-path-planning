# Main script to solve the UV Light optimization problem

import pandas as pd
from room import Room
from polygon_extraction import extract_polygon, construct_isValidLocation_function
from lp_solver import solve_full_lp, visualize_times, solve_naive, visualize_energy, visualize_distance
from shapely.geometry import box
import matplotlib.pyplot as plt
from shapely.ops import transform
import sys

# Quick modification to parse command line arguments
# TODO: Rewrite using argparse
# Format: python main.py USE_NAIVE USE_WEAK USE_STRONG USE_B&B ROOM-ROBOT_EPSILON VISUALIZE
#                  0         1       2           3        4           5             6

# Example: python main.py False False True False 0.3 False

######################
###   Parameters   ###
######################

# I/O Files
INPUT_FILE = '../floor_plans/hrilab_2510_sled.pgm'
INPUT_YAML = '../floor_plans/hrilab_2510_sled.yaml'
OUTPUT_FOLDER = '../output' # Note: No ending '/' in filepath

# Environment parameters
ROBOT_HEIGHT = 1.2192         # Height of UV light, in meters
ROBOT_RADIUS = 0.4            # Distance from robot center to farthest point,
                              # in meters
ROBOT_WATTAGE = 55            # Power of the UV light in Watts (ie. J/sec)
DISINFECTION_THRESHOLD = 1206 # Joules/meter^2

# Preprocessing parameters
ORTHOGONAL_TOL = 20           # Tolerance for orthogonal simplification, in pixels
AVOID_UNKNOWN_REGIONS = True  # Treat "unknown" pixels as walls when determining
                              #  the spaces that the robot can move to


#############################################
###   Loop over all possible parameters   ###
#############################################

def test_all():
    input_files = ['../floor_plans/hrilab_sledbot_twolasers3.pgm',
                   '../floor_plans/2510_centeroffice.pgm',
                   '../floor_plans/2530.pgm',
                   '../floor_plans/2540.pgm',
                   '../floor_plans/2560.pgm',
                   '../floor_plans/2910.pgm']
    input_yamls = [pgm_file[:-3] + 'yaml' for pgm_file in input_files]
    algorithm_combinations = [#('Naive', 1, 0, 0, 0, 'none', 0.2),
                              ('Strong-Visibility', 0, 0, 1, 1, 'none', None),
                              ('Lower-Bound', 0, 1, 0, 0, 'none', None),
                              #('Branch-and-Bound', 0, 0, 0, 0, 'branch_and_bound', None)
                              ]
    epsilons = [1, 0.5, 0.3, 0.2]

    test_from_list(input_files, input_yamls, algorithm_combinations, epsilons)

# algorithm_combinations is a list of tuples representing:
#    (name, naive_solution, use_weak_everything, use_strong_visibility,
#     use_strong_distances, scaling_method, run_with_only_this_epsilon)
def test_from_list(input_files, input_yamls, algorithm_combinations, epsilons):
    SHOW_VISULIZATIONS = False

    output_to_print = "File, Algorithm, Epsilon, Solution Time, Solution Disinfection Percent\n"

    for input_file, input_yaml in zip(input_files, input_yamls):
        for algorithm in algorithm_combinations:
            name, naive_solution, use_weak_everything, use_strong_visibility, use_strong_distances, scaling_method, run_with_only_this_epsilon   = algorithm

            if run_with_only_this_epsilon is None:
                run_with_epsilons = epsilons
            else:
                run_with_epsilons = [run_with_only_this_epsilon]
            
            for epsilon in run_with_epsilons:
                # TODO: Actually grab real name of input in general case...
                #       as opposed to assuming format ../floor_plans/NAME.png
                output_file = "{}/WaitingTimes_{}_{}-{}.csv".format(
                        OUTPUT_FOLDER, input_file[15:-4], name, epsilon)
                print("-"*80)
                print("CURRENTLY RUNNING:", output_file)
                print("-"*80)

                time, percent_disinfected = run_with_parameters(
                                    input_file,
                                    input_yaml,
                                    output_file,
                                    ROBOT_HEIGHT,
                                    ROBOT_RADIUS,
                                    ROBOT_WATTAGE,
                                    DISINFECTION_THRESHOLD,
                                    ORTHOGONAL_TOL,
                                    AVOID_UNKNOWN_REGIONS,
                                    naive_solution,
                                    use_weak_everything,
                                    use_strong_visibility,
                                    use_strong_distances,
                                    scaling_method,
                                    epsilon,
                                    epsilon,
                                    SHOW_VISULIZATIONS)
                curr_output = "{}, {}, {}, {}, {}\n".format(
                        input_file, name, epsilon, time, percent_disinfected)
                output_to_print += curr_output


    print('-'*80)
    print(output_to_print)
    print('-'*80)

                    


#####################################################
###   Compute Solution With Specific Parameters   ###
#####################################################

def run_with_parameters(input_file, input_yaml, output_csv, robot_height, robot_radius,
                        robot_wattage, disinfection_threshold, orthogonal_tol,
                        avoid_unknown_regions, naive_solution, use_weak_everything,
                        use_strong_visibility, use_strong_distances, scaling_method,
                        robot_epsilon, room_epsilon, show_visualizations):

    # Step 1: read input file (pixel-like image) and transform it to a simple polygon
    #         (with clearly marked in/out)
    print('Extracting polygon')
    polygon_data = extract_polygon(input_file,
                                   input_yaml,
                                   ortho_tolerance = orthogonal_tol,
                                   show_visualization = show_visualizations)
    polygon, gray_img, xy_to_pixel, meters_per_pixel = polygon_data
    
    is_valid_location = construct_isValidLocation_function(gray_img,
                                                           xy_to_pixel,
                                                           robot_radius,
                                                           meters_per_pixel,
                                                           avoid_unknown_regions)
    
    # Step 2: a Room object contains not only the boundary, but creates a
    #         discretized list of places for the robot to guard (and list
    #         of places where the robot can actually move to)
    print('Creating room')
    room = Room(polygon,
                gray_img,
                xy_to_pixel,
                robot_buffer_meters = robot_radius,
                is_valid_guard = is_valid_location,
                room_eps = room_epsilon,
                guard_eps = robot_epsilon,
                show_visualization = show_visualizations)
    
    if naive_solution:
        solve_naive(room, robot_height, disinfection_threshold)
    else:
        # Step 3: we generate the LP problem and solve it.
        print('Solving lp')
        lp_solution_data = solve_full_lp(room,
                                         robot_height,
                                         robot_radius,
                                         robot_wattage,
                                         use_weak_everything,
                                         use_strong_visibility,
                                         use_strong_distances,
                                         scaling_method,
                                         disinfection_threshold,
                                         show_visualizations)
        time, waiting_times, intensities, unguarded_room_idx, percent_disinfected = lp_solution_data
    
        # Step 4: Output a solution
        print('Total solution time:', time)
        print('Percent Disinfected:', percent_disinfected)
    
        # Create a csv of all positions and waiting time
        rows = []
        for (x, y), t in zip(room.guard_grid, waiting_times):
            # We drop points that you stop less than a milisecond. HARDCODED
            if t > 1e-3:
                rows.append({'x': x, 'y': y, 'time': t})
        pd.DataFrame(rows).to_csv(output_csv, index=False)
    
        # Graphical visualizations of the solution
        if show_visualizations:
            print('Visualizing solution')
            visualize_times(room, waiting_times, unguarded_room_idx)
            visualize_energy(room, waiting_times, intensities, disinfection_threshold)
            visualize_distance(room, waiting_times, intensities)

        return time, percent_disinfected



if __name__ == '__main__':
    test_all()