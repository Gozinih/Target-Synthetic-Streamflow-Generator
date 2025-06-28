import numpy as np
from tqdm import tqdm

from a8_BuildFeasibleAreaPolygon_and_CheckFeasibility import build_all_polygons, check_feasibility_from_polygons

def adjust_scenario_to_feasible(desired_scenarios_monthly, meanseasonality_change,
                                SDseasonality_change, mean_scenario_range,
                                x_boundary, y_boundary,
                                numberoflocations, isLocal, desired_scenarios_monthly_1):
    """
    Adjust all scenario vectors in partially infeasible scenarios, to ensure feasibility.

    Args:
        desired_scenarios_monthly: monthly mean/sd target deviations
        meanseasonality_change: monthly mean target seasonality
        SDseasonality_change: monthly sd target seasonality
        mean_scenario_range: Allowed mean forcing change [min, max]
        x_boundary : mean change boundaries [n_scenarios x 12 x n_locations]
        y_boundary : SD change boundaries [n_scenarios x 12 x n_locations]
        numberoflocations: Total number of locations
        isLocal: 2D binary array [n_locations x 1] indicating if location is local (1) or non-local (0)

    Returns:
        adjusted_scenarios: adjusted final scenarios which are all feasible [n_scenarios x 24 x numberoflocations], 
    """
    # === Number of scenarios ===
    n_scenarios = desired_scenarios_monthly.shape[0]

    # === Identify non-local location indices ===
    non_local_indices = np.where(isLocal.flatten() == 0)[0]

    # === Initialize output array for all locations (local left as 0) ===
    adjusted_scenarios = np.zeros((n_scenarios, 24, numberoflocations))  # [scenarios x (12 mean + 12 sd) x locations]

    # === Build polygon dictionary (once for all) ===
    polygon_dict = build_all_polygons(mean_scenario_range, x_boundary, y_boundary,desired_scenarios_monthly_1)

    # === Progress bar for non-local adjustments ===
    progress_bar = tqdm(total=n_scenarios, desc="🔍 Adjusting Partially Infeasible Scenarios")

    for sce_idx in range(n_scenarios):
        for k in non_local_indices:  # non-local locations only
            scenario_row = desired_scenarios_monthly[sce_idx, :].copy()
            change = np.zeros(2)

            for j in range(12):  # months
                # === Apply seasonal adjustment to (mean, SD) ===
                change[0] = scenario_row[j] + meanseasonality_change[k, j] + 0.01 * scenario_row[j] * meanseasonality_change[k, j]
                change[1] = scenario_row[j+12] + SDseasonality_change[k, j] + 0.01 * scenario_row[j+12] * SDseasonality_change[k, j]

                # === Clip to lower bound ===
                change[0] = max(change[0], -95)
                change[1] = max(change[1], -95)

                # === Check if point is outside polygon ===
                polygon = polygon_dict[(k, j)]
                x_poly, y_poly = polygon.vertices[:, 0], polygon.vertices[:, 1]
                outside = check_feasibility_from_polygons(change, k, j, polygon_dict)

                if outside:
                    # Find feasible boundary points with mean < target
                    valid_indices = np.where(x_poly < change[0])[0]
                    if valid_indices.size > 0:
                        valid_y = y_poly[valid_indices]
                        closest_idx = valid_indices[np.argmin(np.abs(valid_y - change[1]))]
                    else:
                        closest_idx = np.argmin(np.abs(y_poly - change[1]))

                    adjusted_mean = (x_poly[closest_idx] - meanseasonality_change[k, j]) / (1 + 0.01 * meanseasonality_change[k, j])
                    adjusted_sd = (max(y_poly[closest_idx], -95) - SDseasonality_change[k, j]) / (1 + 0.01 * SDseasonality_change[k, j])

                    adjusted_scenarios[sce_idx, j, k] = adjusted_mean
                    adjusted_scenarios[sce_idx, j+12, k] = adjusted_sd

                else:
                    # Back-solve to ensure correct storage of original scenario (after adjustment)
                    adjusted_mean = (change[0] - meanseasonality_change[k, j]) / (1 + 0.01 * meanseasonality_change[k, j])
                    adjusted_sd = (change[1] - SDseasonality_change[k, j]) / (1 + 0.01 * SDseasonality_change[k, j])

                    adjusted_scenarios[sce_idx, j, k] = adjusted_mean
                    adjusted_scenarios[sce_idx, j+12, k] = adjusted_sd

        progress_bar.update(1)

    progress_bar.close()

    return adjusted_scenarios