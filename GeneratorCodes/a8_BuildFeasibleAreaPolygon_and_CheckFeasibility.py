# a8_BuildFeasibleAreaPolygon_and_CheckFeasibility.py

import numpy as np
from matplotlib.path import Path

"""
Module: Feasibility Check in Exposure Space

This module supports scenario filtering by checking whether a proposed
mean/SD change lies within the feasible region for each month and location.

Key Functions:
    - build_all_polygons(): Builds polygon form generated boundaries in a3_
    - check_feasibility_from_polygons(): Checks if a change vector lies within the polygon
"""


def build_all_polygons(mean_scenario_range, x_boundary, y_boundary, desired_scenarios_monthly_1):
    """
    Constructs 2D feasibility polygons for each (location, month)

    Parameters:
        mean_scenario_range : range used for bounding polygons
        x_boundary : mean change boundaries [n_scenarios x 12 x n_locations]
        y_boundary : SD change boundaries [n_scenarios x 12 x n_locations]
        desired_scenarios_monthly_1 : scenarios to be checked for feasibility

    Returns:
        polygon_dict : A dictionary with keys as (location_index, month_index) and values as matplotlib Path objects.
            Each Path represents the polygonal feasible region in (mean%, sd%) space.
    """
    n_scenarios, _ = desired_scenarios_monthly_1.shape
    _, n_months, n_locations = x_boundary.shape
    polygon_dict = {}

    for loc in range(n_locations):
        for month in range(n_months):
            x_boundary1 = []
            y_boundary1 = []

            # Collect valid (x, y) points with synthetic responses above minimum threshold
            for sce in range(n_scenarios):
                x = x_boundary[sce, month, loc]
                y = y_boundary[sce, month, loc]
                if x >= -95 and y >= -95:  # Ignore values considered invalid or out-of-bound
                    x_boundary1.append(x)
                    y_boundary1.append(y)

            # Add fixed bounding points to ensure closed polygon region
            x_poly = [-95] + x_boundary1 + [mean_scenario_range[1]]
            y_poly = [-95] + y_boundary1 + [-95]

            # Construct polygon and store in dictionary keyed by (location, month)
            polygon = Path(np.column_stack((x_poly, y_poly)))
            polygon_dict[(loc, month)] = polygon

    return polygon_dict


def check_feasibility_from_polygons(change, loc, month, polygon_dict):
    """
    Tests whether a given (mean%, sd%) change lies inside the polygonal feasible region.

    Parameters:
        change : meand and sd change to be checked
        loc : location number
        month :
        polygon_dict : Dictionary of prebuilt polygon boundaries (from build_all_polygons)

    Returns:
        outside : True if the point is outside the polygon (i.e., infeasible)
            False if inside or on the polygon boundary
    """
    polygon = polygon_dict[(loc, month)]
    return not polygon.contains_point((change[0], change[1]))