import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon, Point, LineString, MultiPolygon
from room import Room

# Branch and bound algorithm: divide the rectangle up into subrectangles,
# splitting along the longest side. Lower/upper bound the illumination
# at each rectangle by determining illumination at the corners and taking
# min/max illuminations.
def box_intersection(room, bounds):
    inter = box(*bounds).intersection(room.room)
    #fig, ax = plt.subplots()
    #ax.scatter(room.guard_grid[:, 0], room.guard_grid[:, 1], marker='.')
    #ax.plot(*box(*bounds).exterior.coords.xy, color='red', linestyle='dotted')
    #ax.plot(*room.room.exterior.coords.xy, color='gray', linestyle='--')
    if isinstance(inter, Polygon) and not inter.is_empty:
        #ax.plot(*inter.exterior.coords.xy)
        #plt.show()
        return inter, inter.exterior.coords.xy
    elif isinstance(inter, MultiPolygon) and not inter.is_empty:
        xs, ys = inter[0].exterior.coords.xy
        #ax.plot(xs, ys)
        for part in inter[1:]:
            part_x, part_y = part.exterior.coords.xy
            #ax.plot(xs, ys)
            xs = np.append(xs, part_x)
            ys = np.append(ys, part_y)
        #plt.show()
        return inter, (xs, ys)
    return None, None


def box_sightlines(room, exterior_coords):
    sightlines = np.zeros((len(exterior_coords[0]), room.guard_grid.shape[0]))
    for ext_idx, (x, y) in enumerate(zip(*exterior_coords)):
        for guard_idx, guard_point in enumerate(room.guard_grid):
            sight = LineString([guard_point, (x, y)])
            if room.room.contains(sight):
                sightlines[ext_idx, guard_idx] = 1
    return sightlines
    

def box_intensities(room, weights, exterior_coords):
    box_x, box_y = exterior_coords
    gx = room.guard_grid[:, 0]
    gy = room.guard_grid[:, 1]
    intensities = np.zeros((len(box_x), room.guard_grid.shape[0]))
    for idx, (x, y) in enumerate(zip(box_x, box_y)):
        intensities[idx] = weights / (((x - gx) ** 2) + ((y - gy) ** 2) + height)
    return intensities

def illumination_lower_bound(room, weights, sightlines, exterior_coords):
    intensities = box_intensities(room, weights, exterior_coords)
    intensities[sightlines == 0] = np.inf
    min_intensities = np.min(intensities, axis=0)
    min_intensities[min_intensities == np.inf] = 0
    return np.sum(min_intensities)

def illumination_upper_bound(room, weights, sightlines, poly):
    distances = np.zeros(room.guard_grid.shape[0])
    for idx, (x, y) in enumerate(room.guard_grid):
        if poly.contains(Point(x, y)):
            # Case: guard point within box.
            distances[idx] = 0
        else:
            if isinstance(poly, MultiPolygon):
                distances[idx] = min(p.exterior.distance(Point(x, y))
                                     for p in poly)**2
            else:
                 distances[idx] = poly.exterior.distance(Point(x, y))**2
    distances += height
    return np.sum(np.multiply(weights / distances,
                  np.max(sightlines, axis=0)))

def branch_bound_poly(room, weights, max_iters=15, epsilon=1e-5):
    rects = [room.room.bounds]
    outer_poly, outer_points = box_intersection(room, rects[0])
    outer_sightlines = box_sightlines(room, outer_points)
    lbs = [illumination_lower_bound(room, weights,
                                    outer_sightlines, outer_points)]
    ubs = [illumination_upper_bound(room, weights,
                                    outer_sightlines, outer_poly)]
    L_idx = 0
    Q = rects[0]
    L = lbs[0]
    U = ubs[0]
    
    for i in range(max_iters):
        print(i, L, U, U - L)
        if U - L <= epsilon:
            break
        # Split Q along its longest side and replace with Q1, Q2.
        minx, miny, maxx, maxy = Q
        width = maxx - minx
        height = maxy - miny
        if width > height:
            Q1 = (minx, miny, minx + (maxx - minx) / 2, maxy)
            Q2 = (minx + (maxx - minx) / 2, miny, maxx, maxy)
        else:
            Q1 = (minx, miny, maxx, miny + (maxy - miny) / 2)
            Q2 = (minx, miny + (maxy - miny) / 2, maxx, maxy)
            
        # We have created two new boxes. Inductively, one of them
        # should intersect with the room polygon, but the other
        # may not. We remove this box entirely.
        Q1_poly, Q1_points = box_intersection(room, Q1)
        Q2_poly, Q2_points = box_intersection(room, Q2)
        if not Q1_poly:
            Q2_sightlines = box_sightlines(room, Q2_points)
            rects[L_idx] = Q2
            lbs[L_idx] = illumination_lower_bound(room, weights,
                                                  Q2_sightlines, Q2_points)
            ubs[L_idx] = illumination_upper_bound(room, weights,
                                                  Q2_sightlines, Q2_poly)
        else:
            Q1_sightlines = box_sightlines(room, Q1_points)
            rects[L_idx] = Q1
            lbs[L_idx] = illumination_lower_bound(room, weights,
                                                  Q1_sightlines, Q1_points)
            ubs[L_idx] = illumination_upper_bound(room, weights,
                                                  Q1_sightlines, Q1_poly)
            if Q2_poly:
                Q2_sightlines = box_sightlines(room, Q2_points)
                rects.append(Q2)
                lbs.append(illumination_lower_bound(room, weights,
                                                    Q2_sightlines, Q2_points))
                ubs.append(illumination_upper_bound(room, weights,
                                                     Q2_sightlines, Q2_poly))
                
        # Update bounds.
        L_idx = np.argmin(lbs)
        Q = rects[L_idx]
        L = np.min(lbs)
        U = np.min(ubs)
    return Q, L, U