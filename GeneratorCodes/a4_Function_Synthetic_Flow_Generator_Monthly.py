# a4_Function_Synthetic_Flow_Generator_Monthly.py

# Developed based on the synthetic streamflow generator by Kirsch et al. (2013):
# Kirsch, B. R., Characklis, G. W., & Zeff, H. B. (2013). Evaluating the Impact of 
# Alternative Hydro-Climate Scenarios on Transfer Agreements: Practical Improvement 
# for Generating Synthetic Streamflows. Journal of Water Resources Planning and Management, 
# 139(4), 396–406. https://doi.org/10.1061/(asce)wr.1943-5452.0000287

import numpy as np
from numba import njit

@njit
def compute_monthly_aggregates(data1, firstyear, lastyear):
    """
    Aggregate daily streamflow into monthly totals.

    Parameters:
        data1: full historical dataset (daily inflow columns by location).
        firstyear: first year in the dataset
        lastyear: last year in the dataset

    Returns:
        inputdata: ndarray of shape with monthly totals
    """
    numberofyears_recorded = lastyear - firstyear + 1
    inputdata = np.zeros((numberofyears_recorded, 12))
    for i in range(data1.shape[0]):
        y = int(data1[i, 0])
        m = int(data1[i, 1])
        if firstyear <= y <= lastyear and 1 <= m <= 12:
            inputdata[y - firstyear, m - 1] += data1[i, 2]
    return inputdata

@njit
def fill_synthetic_uncorrelated(standardized_inputdata, randomyear, firstyear):
    """
    Build synthetic matrix by drawing standardized values based on random year matrix.

    Parameters:
        standardized_inputdata: standardized historical data
        randomyear: matrix of sampled years
        firstyear: start of historical period

    Returns:
        syntheticdata_uncorrelated: sampled synthetic matrix
    """
    n_synth = randomyear.shape[0]
    syntheticdata_uncorrelated = np.zeros((n_synth, 12))
    for i in range(n_synth):
        for j in range(12):
            syntheticdata_uncorrelated[i, j] = standardized_inputdata[randomyear[i, j] - firstyear, j]
    return syntheticdata_uncorrelated

@njit
def reshape_standardized(input_data):
    """
    Reshape matrix to create overlapping segments for considering correlation between years (Kirsch et al. strategy).

    Parameters:
        input_data: matrix [numberofyears_recorded x 12]

    Returns:
        reshaped: matrix [numberofyears_recorded - 1 x 12]
    """
    n = input_data.shape[0] - 1
    reshaped = np.zeros((n, 12))
    for i in range(n):
        for j in range(12):
            if j < 6:
                reshaped[i, j] = input_data[i, j + 6]
            else:
                reshaped[i, j] = input_data[i + 1, j - 6]
    return reshaped

@njit
def compute_ln_params(meanchange, SDchange, mean_monthly, SD_monthly):
    """
    Convert percent mean/SD change to log-space adjustment factors.

    Parameters:
        meanchange, SDchange: monthly forcing percent changes in real space
        mean_monthly, SD_monthly: recorded mean and sd in log-space

    Returns:
        meanchange_ln, SDchange_ln: monthly forcing percent changes in log space
    """
    meanchange_factors = 1 + meanchange / 100
    SDchange_factors = 1 + SDchange / 100
    meanchange_ln = np.ones(12)
    SDchange_ln = np.ones(12)

    for j in range(12):
        A = np.exp(SD_monthly[j] ** 2) - 1
        B = (SDchange_factors[j] / meanchange_factors[j]) ** 2
        SDchange_ln[j] = np.sqrt(np.log(B * A + 1)) / SD_monthly[j]
        meanchange_ln[j] = (
            (np.log(meanchange_factors[j]) + mean_monthly[j] + (SD_monthly[j] ** 2) / 2
             - 0.5 * np.log(B * A + 1)) / mean_monthly[j])
    return meanchange_ln, SDchange_ln

@njit
def de_standardize(synth_corr_combined, mean_monthly, SD_monthly, mean_ln, sd_ln):
    """
    Reverse standardization and log-transform to obtain real-space flows.

    Parameters:
        synth_corr_combined: synthetic standardized correlated data
        mean_monthly, SD_monthly: recorded mean and sd in log-space
        mean_ln, sd_ln: monthly forcing percent changes in log space

    Returns:
        synthetic_real: de-standardized synthetic flows in real space
    """
    n_years = synth_corr_combined.shape[0]
    synthetic_real = np.zeros_like(synth_corr_combined)
    for i in range(n_years):
        for j in range(12):
            synthetic_real[i, j] = np.exp(
                synth_corr_combined[i, j] * SD_monthly[j] * sd_ln[j] +
                mean_monthly[j] * mean_ln[j])
    return synthetic_real

def synthetic_flow_generator_monthly(inflowdata, locationnumber, numberofyears_syntheticdata,
                                     randomyear, meanchange, SDchange):
    """
    Main function to generate synthetic monthly streamflow based on the forcing mean and sd changes.

    Parameters:
        inflowdata: full recorded data
        locationnumber: location number
        numberofyears_syntheticdata: synthetic years to generate (including extra year)
        randomyear: random matrix year
        meanchange, SDchange: forcing percent changes in mean and SD in real space

    Returns:
        synthetic_real: generated synthetic streamflow
        inputdata: aggregated monthly data from historical series
    """
    # === Step 0: Extract year, month, and location flow ===
    data1 = inflowdata[:, [1, 2, locationnumber + 3]].astype(np.float64)
    firstyear = int(data1[0, 0])
    lastyear = int(data1[-1, 0])

    # === Step 1: Aggregate historical daily to monthly flow ===
    inputdata = compute_monthly_aggregates(data1, firstyear, lastyear)

    # === Step 2: Apply log transformation and standardize ===
    ln_inputdata = np.log(inputdata)
    mean_monthly = np.mean(ln_inputdata, axis=0)
    SD_monthly = np.std(ln_inputdata, axis=0, ddof=1)
    standardized_inputdata = (ln_inputdata - mean_monthly) / SD_monthly

    # === Step 3: Generate synthetic flows using year resampling ===
    syntheticdata_uncorrelated = fill_synthetic_uncorrelated(standardized_inputdata, randomyear, firstyear)

    # === Step 4: Apply correlation structure from historical log flows ===
    auto_corr = np.corrcoef(standardized_inputdata, rowvar=False)
    upper_triangle = np.linalg.cholesky(auto_corr).T
    syntheticdata_correlated = syntheticdata_uncorrelated @ upper_triangle

    # === Step 5: Edge correction — reshape & re-apply correlation ===
    standardized_inputdata_2 = reshape_standardized(standardized_inputdata)
    syntheticdata_uncorrelated_2 = reshape_standardized(syntheticdata_uncorrelated)

    auto_corr_2 = np.corrcoef(standardized_inputdata_2, rowvar=False)
    upper_triangle_2 = np.linalg.cholesky(auto_corr_2).T
    syntheticdata_correlated_2 = syntheticdata_uncorrelated_2 @ upper_triangle_2

    # === Step 6: Combine halves from staggered correlation versions ===
    syntheticdata_correlated_combined = np.hstack([
        syntheticdata_correlated[1:numberofyears_syntheticdata, 6:12],
        syntheticdata_correlated_2[:, 6:12]
    ])

    # === Step 7: Back-transform to real-space synthetic flows ===
    mean_ln, sd_ln = compute_ln_params(meanchange, SDchange, mean_monthly, SD_monthly)
    synthetic_real = de_standardize(syntheticdata_correlated_combined, mean_monthly, SD_monthly, mean_ln, sd_ln)

    return synthetic_real, inputdata