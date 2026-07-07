# pantheon-directional-analysis

**Covariance-aware directional analysis of the Pantheon+ Type Ia supernova sample**

This repository contains the complete code and data required to reproduce the results of the paper:

> **A robust directional Hubble analysis of Pantheon+: a low-redshift dipole-like signal consistent with local coherent flow**

The analysis implements a statistically controlled framework to test for anisotropy in the Hubble diagram, combining:
- Generalized least-squares fitting with full covariance whitening
- Empirical null calibration from 100,000 isotropic simulations
- Global look-elsewhere correction
- Multipole leakage estimation and injection-recovery tests
- Robustness diagnostics (jackknife, hemisphere splits, survey splits)
- Cumulative redshift tomography
- Bulk-flow fitting and Shapley-cone removal tests

---

## 📥 Data Setup

This repository includes **two** file formats for the Pantheon+ dataset. Please pay attention to which one you use:

- **`pantheon_data.zip`** – **Recommended for analysis.**  
  This archive contains all required data files (`Pantheon+SH0ES.dat` and `Pantheon+SH0ES_STAT+SYS.cov.txt`).  
  After downloading, **extract the ZIP** into the project root directory. The code will automatically find and load the data.

- **`pantheon_data.mht`** – Web archive format (for reference only).  
  ⚠️ **Do not use this file** for running the analysis – it is not compatible with the pipeline.

---

## 🚀 Quick Start (Run the analysis)

1. **Download the ZIP file**  
   Click on `pantheon_data.zip` in this repository and download it.

2. **Extract the archive**  
   Unzip the file into the same folder as the Python script.

3. **Install dependencies** (if not already installed):  
   ```bash
   pip install numpy scipy
