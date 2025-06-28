# Target Synthetic Streamflow Generator

This repository provides a Python-based framework to generate **target synthetic streamflow scenarios**, targeting monthly magnitude, variability, and seasonality, while ensuring feasibility based on historical data.

The project is a faithful translation and improvement of an original MATLAB implementation, helped by ChatGPT, designed to support water resources planning under different climate or policy change scenarios.

The methodology of **Nowak et al. (2010)** is used for disaggregation and **Kirsch et al. (2013)** for synthetic flow generation. 
See [📚 References](#-references) for full citations.

---

## 📦 Key Features

- Generate synthetic streamflow scenarios, targeting monthly magnitude (mean), variation (standard deviation), and seasonality.
- Remove fully infeasible scenarios and adjust partially infeasible ones, based on boundary scenarios.
- Optimize monthly forcing values to match target changes.
- Convert optimized monthly flows into daily time series (with leap year correction).
- Supports **local** (resampling) and **non-local** (optimization) stations.
- Resampling for local locations would be from historical data with the same recorded sequence.
- Built-in visualization tools to plot exposure space, flow duration curves, and monthly time series.
- Includes example data for a 14-inflow-location case study to test and validate the generator.

---

## 💻 **Recommended Development Environment**:  
The codebase was developed and tested using [Visual Studio Code](https://code.visualstudio.com/) with the Python 3.11 interpreter. 
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue) 
The **H5Web** extension is useful for exploring `.h5` output files directly within VS Code.

```

## 📁 Repository Structure

```
.
├── GeneratorCodes/
│   ├── Boundary                    # Saved Boundary Scenarios.
│   ├── a1_Main.py                  # Main pipeline
│   ├── a2_... to a17_...py         # Modular components (boundary generation, optimization, disaggregation, etc.)
├── PlottingCodes/                  # Visualization tools for analyzing scenario results
│   ├── c1_.py                      # Plots exposure space (mean vs SD) for selected locations
│   ├── c2_.py                      # Flow Duration Curves: synthetic vs. historical
│   ├── c3_.py                      # Time series plots: synthetic vs. historical
├── Scenarios                       # Generated Scenarios will be saved here
├── InputData.txt                   # Editable input
├── LICENSE 
├── Mean_Seasonality.xlsx           # Editable input 
├── RandomYearMatrix.xlsx           # Editable input
├── README.md
├── RandomYearMatrix.xlsx           # Editable input
├── Run.bat                         # .bat file to Run                      
├── Run.py                          # Entry point for batch execution
└── SD_Seasonality.xlsx             # Editable input
```

---

## 🧠 Workflow Overview

1. **Input Configuration**  
   Set your simulation parameters and file paths in `InputData.txt`.
   `RecordedData.xlsx`: Daily flow data and `isLocal` flags (2 sheets)
   `Mean_Seasonality.xlsx`: Seasonal mean change matrix [locations × 12]
   `SD_Seasonality.xlsx`: Seasonal SD change matrix [locations × 12]
   *(Optional)* `RandomYearMatrix.xlsx`: Year sampling matrix (auto-generated if missing)

2. **Execution**  
   Run the full process via:
   ```bash
   python Run.py
   ```

3. **Outputs**  
   Depending on the flags, the script will produce:
   - Daily (cms) CSV files (`/DailyTimeseriesCSVFiles`)
   - Monthly (cms) CSV files (`/MonthlyTimeseriesCSVFiles`)
   - HDF5 files with all results (`/OutputData`)
   - Saved HDF5 input for reproducibility (`/InputData`)
   > 💡 To browse `.h5` files easily in VS Code, install the **H5Web** extension from the Extensions Marketplace.

---

## 📊 Scenario Generation Logic

The process includes:
- Random year matrix generation (`a2`)
- Boundary space generation (`a3`, `a4`, `a5`, `a6`)
- Stochastic streamflow generator  (`a4`) **Kirsch et al. (2013)**
- Mean and SD calculator functions (`a5`, `a6`)
- Feasibility checks and adjustments (`a7`, `a8`, `a9`)
- Inverse optimization + disaggregation (`a10` to `a16`)
- Disaggregation from monthly to daily (`a17`) **Nowak et al. (2010)**

Each script is modular, documented, and uses Numba-accelerated routines for performance.

---

## 📈 Visualization Tools (`PlottingCodes/`)

The repository includes optional plotting scripts for scenario evaluation:

- **c1_ExposureSpacePlot.py**  
  Visualizes the exposure space (mean and SD changes) for any two selected locations across all months and scenarios. Useful for checking feasibility and alignment with target zones.

- **c2_FlowDurationCurve.py**  
  Plots monthly Flow Duration Curves (FDCs) comparing synthetic scenarios against historical flow records, for 4 selected locations. Useful for validating statistical flow distributions.

- **c3_TimeSeries_MonthlyFlow.py**  
  Displays time series plots of monthly flow for 4 selected locations. Overlays recorded flow data with the full synthetic min–max envelope.

Each plotting tool reads `.h5` scenario outputs from `/Scenarios/OutputData` and is customizable to focus on different locations.

---

## ⚙️ Dependencies

Developed and tested with **Python 3.11**
Install required Python packages (automatically handled by `Run.py`):
- `numpy`, `pandas`, `scipy`, `h5py`, `joblib`, `openpyxl`, `numba`, `matplotlib`, `numba`

---

## 📚 References

- **a17_Disaggregation.py**  
  Based on the disaggregation method from:  
  Nowak, K., Prairie, J., Rajagopalan, B., & Lall, U. (2010).  
  *A nonparametric stochastic approach for multisite disaggregation of annual to daily streamflow.*  
  *Water Resources Research, 46(8).*  
  [https://doi.org/10.1029/2009WR008530](https://doi.org/10.1029/2009WR008530)

- **a4_Function_Synthetic_Flow_Generator_Monthly.py**  
  Based on the synthetic streamflow generator by:  
  Kirsch, B. R., Characklis, G. W., & Zeff, H. B. (2013).  
  *Evaluating the Impact of Alternative Hydro-Climate Scenarios on Transfer Agreements: Practical Improvement for Generating Synthetic Streamflows.*  
  *Journal of Water Resources Planning and Management, 139(4), 396–406.*  
  [https://doi.org/10.1061/(asce)wr.1943-5452.0000287](https://doi.org/10.1061/(asce)wr.1943-5452.0000287)

## 📜 License

See [LICENSE](./LICENSE) for licensing terms.

---

## ✍️ Citation

If you use this code in research or policy applications, please cite the original authors and contributors. Attribution is appreciated when extending or modifying the code.

---

## 🙋 Contact

For questions or suggestions, feel free to open an issue or submit a pull request.
hamid.gozini@gmail.com
gozinih@myumanitoba.ca
