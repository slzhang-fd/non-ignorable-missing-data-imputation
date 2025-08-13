# Non-ignorable Missing Data Imputation

## Description

This repository contains the complete replication code for the research article:

> **Imputation Analysis of Large-Scale Mixed Data with Non-ignorable Missingness: A Latent Variable Modeling Approach**

The codebase implements novel latent variable modeling approaches for handling non-ignorable missingness in mixed-type datasets (continuous, binary, and ordinal variables). All simulation studies presented in the paper can be reproduced using this code.


## Repository Structure

### Simulation Studies
- **Simulation 1** (`simu1/`): Comparison of ignorable vs non-ignorable approaches
  - `simu1-1.py` - Ignorable true model with ignorable imputation
  - `simu1-2.py` - Non-ignorable true model with ignorable imputation
  - `simu1-*-summary.py` - Results summarization scripts
  
- **Simulation 2** (`simu2/`): Non-ignorable imputation performance
  - `simu2-1.py`, `simu2-2.py` - Different non-ignorable configurations
  - `simu2-*-summary.py` - Results summarization scripts

- **Simulation 3** (`simu3/`): Model robustness testing
  - `simu3.py` - Main simulation with graphical model data
  - `simu_graphical_depends.py` - Graphical model utilities
  - `gen_params.py` - Parameter generation
  - `params.pkl` - Precomputed simulation parameters

### Real Data Analysis
- `real_data/real_analysis.py` - European Social Survey (ESS) Round 9 analysis code
- Analysis of justice and fairness perceptions across 29 European countries
- **Note**: Due to privacy restrictions, the actual ESS data cannot be shared. Only the analysis source code is provided for methodological reference.

### Results
- `res_folder/` - Simulation and analysis results storage
- Organized by study with parameter estimates and convergence diagnostics

## Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- polyagamma (for Pólya-Gamma random variables)
- pandas (for real data analysis)
- matplotlib, seaborn (for visualization)
- SLURM environment (recommended for large-scale replications)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd non-ignorable-missing-data-imputation

# Install required packages
pip install torch numpy polyagamma pandas matplotlib seaborn
```

## Usage

### Running Individual Simulations

The simulation scripts are configured to run standalone with a single replication (`reps = 1`):

```bash
# Simulation Study 1
python simu1/simu1-1.py
python simu1/simu1-2.py

# Simulation Study 2  
python simu2/simu2-1.py
python simu2/simu2-2.py

# Simulation Study 3
python simu3/simu3.py

# Real data analysis (source code only - requires ESS data)
# python real_data/real_analysis.py
```

### Running with SLURM (Recommended for Full Replication)

For complete replication with multiple replications, uncomment the SLURM_ARRAY_TASK_ID lines in each script and run:

```bash
# Submit job array for 100 replications
sbatch --array=1-100 run.sh
```

The `run.sh` script is configured for SLURM environments and will automatically handle replication indexing via `SLURM_ARRAY_TASK_ID`. Each script contains commented lines that can be uncommented to switch from standalone mode to SLURM array mode.

### Generating Summary Results

After running simulations, generate summaries:

```bash
python simu1/simu1-1-summary.py
python simu1/simu1-2-summary.py
python simu2/simu2-1-summary.py
python simu2/simu2-2-summary.py
python simu3/simu3-summary.py
```

## Reproducibility

All simulations use fixed random seeds for reproducibility. The `SLURM_ARRAY_TASK_ID` environment variable controls replication indexing in HPC environments.

Results are stored in pickle format within `res_folder/` and can be analyzed using the provided summary scripts.


## License

This project is licensed under the GPL v3.0 License.
