# README for Protocol to Analyze 1D and 2D Mass Spectrometry Data for Cancer Diagnosis and Immune Cell Identification

## Overview

In the context of cancer diagnosis using mass spectrometry (MS), creating accurate and reliable classification models is crucial. Our pipeline leverages both supervised and unsupervised machine learning approaches to achieve high accuracy in classification and to identify robust biomarkers. We also explore immune cell infiltration in tissues using MS-imaging data to provide insights into the tumor microenvironment without relying on probe-based techniques.

This repository contains:
- Two CSV files as toy examples for training_validation (`data_train_toy.csv`), and testing datasets (`data_test_toy.csv`)
- A folder named `PRISM_Lib` containing three Python files:
  - `Supervised.py`: Contains functions for supervised learning for the 1D pipeline.
  - `Unsupervised.py`: Contains functions for unsupervised learning for the 1D pipeline.
  - `MSI_immunoscoring.py`: Contains functions for the 2D pipeline for immunoscoring using MS-imaging data.
- A text file named "link_to_download_2DMSI_dataset.txt" with the URL to download an example imzML file for the MSI dataset.
- An example of a pre-trained model on immune cells saved as `Model_immunoscoring_toy.pkl` for testing purposes.

## Before You Begin

### Requirements
- **Mass Spectrometry Data**: Our pipeline is initially developed using SpiderMass MS and MSI data but can be applied to various types of MS data.
- **Dataset Structure**: The training dataset for the 1D pipeline should be in CSV format with the first column as "Class" (labels) and subsequent columns as m/z features.
- The blind validation set should follow the same structure.
- **imzML Format**: For the 2D pipeline, an MSI dataset in imzML format and a pre-trained model is required (a pre-trained model could be obtained using the 1D pipeline).

### Setup

#### Install Anaconda:
1. Visit [Anaconda](https://www.anaconda.com/download) and download the appropriate installer.
2. Follow the installation instructions for your operating system.
3. Create a new environemment 

#### Launch Jupyter Notebook:
1. Open Anaconda-Navigator and launch Jupyter Notebook.

#### Install Required Packages:
1. In Jupyter Notebook, run the following code to install necessary packages:
   ```python
   pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.2.2 matplotlib==3.7.2 scipy==1.11.1 seaborn==0.12.2 statannot==0.2.3 lazypredict==0.2.12 joblib==1.3.1 eli5==0.13.0 pyimzml==1.5.3 plotly==5.17.0 lightgbm


## Pipeline Details
Examples usage:
```python
### 1D Pipeline
## Supervised Learning

from PRISM_Lib.Supervised import Train_model
models = Train_model(data)
print(models)

## Unupervised Learning

from PRISM_Lib.Unsupervised import peak_picking, create_heatmap
data_peak_picked = peak_picking(data)
create_heatmap(data_peak_picked)

### 2D Pipeline

from PRISM_Lib.MSI_immunoscoring import generate_tic_map
imzml_file = "MSI_toy.imzml"  # the .ibd file should be located in the same folder as the imzml file
generate_tic_map(imzml_file, mzs_range=(600, 1000))
```

####Example scripts 
Template_1D_Analysis and Template_2D_Analysis for using the pipeline functions are provided in the repository to guide you through the process.

## References
- Zirem et al.,"Real-time glioblastoma tumor microenvironment assessment by SpiderMass for improved patient management." Cell Reports Medicine 5.4 (2024). DOI:https://doi.org/10.1016/j.xcrm.2024.101482
## Contact
For questions or issues, please contact Yanis Zirem at yanis.ziem16@gmail.com or yanis.zirem@univ-lille.fr.
