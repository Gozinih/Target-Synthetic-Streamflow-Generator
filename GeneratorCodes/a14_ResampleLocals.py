# a14_ResampleLocals.py

import numpy as np
from numba import njit
from a4_Function_Synthetic_Flow_Generator_Monthly import synthetic_flow_generator_monthly

@njit
def build_year_sequence(firstyear, lastyear, numberofyears_syntheticdata):
    """
    Create a cyclically repeating matrix of years for local station resampling.

    Parameters:
        firstyear : first year in the recorded dataset
        lastyear : last year in the recorded dataset
        numberofyears_syntheticdata : number of synthetic years to generate

    Returns:
        randomyear_L : matrix assigning each month to a year
                        (wraps around if the number of years exceeds the historical range)
    """
    randomyear_L = np.zeros((numberofyears_syntheticdata, 12), dtype=np.int32)
    for j in range(12):  # Loop over months
        year_ptr = firstyear
        for i in range(numberofyears_syntheticdata):
            randomyear_L[i, j] = year_ptr
            year_ptr += 1
            if year_ptr > lastyear:
                year_ptr = firstyear  # Wrap around if we exceed the last year
    return randomyear_L

@njit
def resample_monthly_flows(x4, randomyear_L, firstyear):
    """
    Resample synthetic monthly flows using the wrapped year matrix.

    Parameters:
        x4 : recorded monthly flow matrix
        randomyear_L : synthetic year-to-month mapping matrix
        firstyear : base year (used to compute correct row index)

    Returns:
        x3_resampled : synthetic monthly flows created by resampling
    """
    n_years = randomyear_L.shape[0] - 1
    x3_resampled = np.zeros((n_years, 12))
    for i in range(n_years):
        for j in range(12):
            x3_resampled[i, j] = x4[randomyear_L[i, j] - firstyear, j]
    return x3_resampled

def resample_locals(data, k, numberofyears_syntheticdata):
    """
    Generate synthetic monthly flows for a local station by resampling recorded data.

    This is used for local (non-optimized) stations where no scenario forcing is applied.
    The synthetic time series is created by wrapping historical records cyclically.

    Parameters:
        data : full daily recorded dataset
        k : index of the target location
        numberofyears_syntheticdata : number of synthetic years to generate

    Returns:
        x3_resampled : synthetic monthly flow matrix, resampled
        x4 : recorded monthly flow matrix
        dist1 : placeholder, always 0
        mean_change_syn1 : placeholder, always 0
        SD_change_syn1 : placeholder, always 0
    """
    # Step 1: Set zero mean/SD change (no forcing for locals)
    meanchange_L = np.zeros(12)
    SDchange_L = np.zeros(12)

    # Step 2: Extract year range from the input data
    firstyear = int(data[0, 1])
    lastyear = int(data[-1, 1])

    # Step 3: Build cyclic year assignment matrix
    randomyear_L = build_year_sequence(firstyear, lastyear, numberofyears_syntheticdata)

    # Step 4: Generate recorded and synthetic flow matrix (no change applied)
    _, x4 = synthetic_flow_generator_monthly(
        data, k, numberofyears_syntheticdata, randomyear_L,
        meanchange_L, SDchange_L
    )

    # Step 5: Resample using wrapped years (exclude last synthetic year)
    x3_resampled = resample_monthly_flows(x4, randomyear_L, firstyear)

    # Placeholders for compatibility with optimization outputs
    dist1 = 0
    mean_change_syn1 = 0
    SD_change_syn1 = 0

    return x3_resampled, x4, dist1, mean_change_syn1, SD_change_syn1