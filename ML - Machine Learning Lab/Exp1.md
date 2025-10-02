## Setup Instructions

Follow these steps to create a virtual environment and install the necessary libraries for your machine learning project.

### Step 1: Open Anaconda Prompt
- Click **Start** → **Anaconda** → **Anaconda Prompt**.


### Step 2: Create and Activate a Virtual Environment
```bash
conda create -n ml_env python=3.11
conda activate ml_env
```

### Step 3: Install Required Libraries

```bash
conda install numpy scipy scikit-learn pandas matplotlib
```

### Step 4: Verify Installed Versions

Run the following commands to check the installed versions of the libraries:

```bash
python -c "import numpy; print('NumPy Version:', numpy.__version__)"
python -c "import scipy; print('SciPy Version:', scipy.__version__)"
python -c "import sklearn; print('Scikit-learn Version:', sklearn.__version__)"
python -c "import pandas; print('Pandas Version:', pandas.__version__)"
python -c "import matplotlib; print('Matplotlib Version:', matplotlib.__version__)"
```

---

Your environment is now ready for use!

