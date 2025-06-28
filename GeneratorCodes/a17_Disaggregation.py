# a17_Disaggregation.py

# Developed based on the method from Nowak et al. (2010):
# Nowak, K., Prairie, J., Rajagopalan, B., & Lall, U. (2010). A nonparametric 
# stochastic approach for multisite disaggregation of annual to daily streamflow. 
# Water Resources Research, 46(8). https://doi.org/10.1029/2009WR008530 

import numpy as np
from numba import njit

@njit
def build_proportion_matrix(Data, Monthly_Recorded, firstyear, station_indices):
    """
    Build a matrix of daily-to-monthly proportions for each selected station.

    Each daily value is divided by the corresponding monthly total for that station,
    resulting in a proportion (used for rescaling synthetic monthly flows).

    Parameters:
        Data : historical daily data
        Monthly_Recorded : recorded monthly totals
        firstyear : first year in the dataset
        station_indices : list of column indices corresponding to target stations

    Returns:
        proportions : daily/monthly ratio matrix
        years : array of year values per day
        months : array of month values per day
    """
    n_days = Data.shape[0]
    n_stations = len(station_indices)
    years = Data[:, 0].astype(np.int32)
    months = Data[:, 1].astype(np.int32)

    proportions = np.zeros((n_days, n_stations))

    for s_idx in range(n_stations):
        station = station_indices[s_idx]
        for i in range(n_days):
            yr_idx = years[i] - firstyear
            mo = months[i] - 1
            monthly_total = Monthly_Recorded[yr_idx, mo, s_idx]
            if monthly_total > 0:
                proportions[i, s_idx] = Data[i, station + 2] / monthly_total
            else:
                proportions[i, s_idx] = 0.0  # avoid divide-by-zero
    return proportions, years, months

def adjust_february(data, startyear_synthetic):
    """
    Adjusts the synthetic dataset for leap years.

    For each synthetic year:
    - Inserts Feb 29 with averaged flow if it's a leap year and only Feb 28 is present
    - Removes Feb 29 if it's a non-leap year but accidentally included

    Parameters:
        data : synthetic daily data as list or ndarray.
        startyear_synthetic : base calendar year for synthetic data

    Returns:
        Modified daily dataset with leap year consistency enforced
    """
    data = data.tolist()
    i = 0
    feb_count = 0

    while i < len(data):
        y_idx = int(data[i][0])
        m = int(data[i][1])
        year = y_idx + startyear_synthetic - 1

        if m == 2:
            feb_count += 1
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            is_last_feb = (i == len(data) - 1) or (int(data[i + 1][1]) != 2)

            if is_last_feb:
                if is_leap and feb_count == 28:
                    # Insert Feb 29 by copying Feb 28 and halving the values
                    row = data[i].copy()
                    row[2:] = [v / 2 for v in row[2:]]
                    data.insert(i + 1, row)
                    i += 1
                elif not is_leap and feb_count == 29:
                    # Merge Feb 29 into Feb 28 by summing and removing the extra row
                    for j in range(2, len(data[i])):
                        data[i - 1][j] += data[i][j]
                    data.pop(i)
                    i -= 1
                feb_count = 0
        else:
            feb_count = 0
        i += 1

    return np.array(data)

@njit
def disaggregate_monthly_flows(Monthly_Synthetic, years, months, proportions, selected_years):
    """
    Disaggregate synthetic monthly flows into daily flows using proportions.

    Each synthetic monthly value is scaled into daily values using the corresponding
    historical proportions from a selected year.

    Parameters:
        Monthly_Synthetic : synthetic monthly data
        years : daily years vector
        months : daily months vector
        proportions : daily/monthly proportion matrix
        selected_years : selected historical year for each synthetic year

    Returns:
        output : synthetic daily data
    """
    n_stations = proportions.shape[1]
    n_days_total = 0

    # First pass: count how many rows needed
    for i in range(Monthly_Synthetic.shape[0]):
        for j in range(12):
            for d in range(years.shape[0]):
                if years[d] == selected_years[i] and months[d] == j + 1:
                    n_days_total += 1

    # Initialize daily output array
    output = np.zeros((n_days_total, 2 + n_stations), dtype=np.float64)

    row = 0
    for i in range(Monthly_Synthetic.shape[0]):
        for j in range(12):
            for d in range(years.shape[0]):
                if years[d] == selected_years[i] and months[d] == j + 1:
                    output[row, 0] = i + 1         # synthetic year index
                    output[row, 1] = j + 1         # month index
                    for s in range(n_stations):
                        output[row, 2 + s] = Monthly_Synthetic[i, j, s] * proportions[d, s]
                    row += 1

    return output
