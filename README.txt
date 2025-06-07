livetime_model releases

# Project Runtime Environment (Conda-based)

[Python Environment]
- Python Version: 3.11
- Environment Manager: Conda
- Environment Name: ai-env

[Dependencies]
- Installation Method: Mix of conda and pip
- Key Libraries:
  - tensorflow==2.15.0
  - scikit-learn==1.3.0
  - matplotlib==3.9.0
  - numpy==1.26.4
  - transformers==4.51.3

[Environment Setup Instructions]
conda create -n ai-env python=3.11
conda activate ai-env
pip install -r requirements.txt

[Notes]
- This project is designed to run in a Conda environment.
- The `requirements.txt` file is generated via `pip freeze` and may not reflect all conda-installed packages.
- For GPU support, ensure CUDA and cuDNN are properly installed.

[System Information]
- Operating System: Ubuntu 22.04 (tested)
- Tested Python Version: 3.11
- Optional: Conda environment file `environment.yml` is available for full environment replication.
