# a2_MatrixYear.py
import numpy as np
import pandas as pd

def matrix_year(Data, numberofyears_syntheticdata):
    """
     Generate Random Year Matrix for Synthetic Scenarios
     This function generates a fixed matrix of randomly selected years from
     the historical dataset. It is used for assigning monthly sampling years
     in synthetic flow generation.

    Input:
          Data - Full daily recorded streamflow dataset (with year info)
          numberofyears_syntheticdata - Number of synthetic years to generate (original numberofyears_syntheticdata + 1)
    
    Output:
          randomyear - A (numberofyears_syntheticdata x 12) matrix
          Each entry is a randomly chosen year from historical range
    """

    # Extract time span from input data
    firstyear = int(Data[0, 1])
    lastyear = int(Data[-1, 1])
    
    # Generate random year matrix for 12 months over desired synthetic years
    randomyear = np.random.randint(firstyear, lastyear + 1,
                                   size=(numberofyears_syntheticdata, 12))

    # Save randomyear to Excel
    pd.DataFrame(randomyear).to_excel('RandomYearMatrix.xlsx', header=False, index=False)

    return randomyear