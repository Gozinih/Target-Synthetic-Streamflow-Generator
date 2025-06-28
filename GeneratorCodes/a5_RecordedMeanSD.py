# a5_RecordedMeanSD.py
import numpy as np

def recorded_mean_sd(x4):
    """
    Calculates monthly mean and standard deviation of recorded streamflow data.

    Args:
        x4: Matrix of monthly recorded flow for one location

    Returns:
        mean_monthly_recorded, SD_monthly_recorded
    """
    mean_monthly_recorded = np.mean(x4, axis=0)
    SD_monthly_recorded = np.std(x4, axis=0, ddof=1)  # Sample std (like MATLAB)

    return mean_monthly_recorded, SD_monthly_recorded