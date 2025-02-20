# Seven Things to Know about Exercise Monitoring with Inertial Sensing Wearables

## Overview

This repository presents the implementation of our collaborative project between UPenn/Orthopaedics, UDel/Physical Therapy, and CMU/Engineering on exercise prediction. The manuscript reporting outcomes of this project is published on the *IEEE Journal of Biomedical and Health Informatics (JBHI)*. 

The paper is available at [this link](https://doi.org/10.1109/JBHI.2024.3368042).

## Data

Data were collected from 19 participants performing 37 exercises while wearing 10 inertial measurement units (IMUs) on chest, pelvis, wrists, thighs, shanks, and feet (see the figure below).

![figure [exercise]: Algorithm Overview](figure/exercise.png)

You may use data samples in the `data` to run the code, or download the full dataset on [SimTK](https://simtk.org/projects/imu-exercise).

## Directory 

You may preserve the following directory tree to run the code on your local machine without further modifications.

```
$ Directory tree
.
├── data\
│   ├── parsed_h5_csv
│   │   └── (IMU data here)
│   └── parsed_joint_angles_all
│       └── (joint angles data here)
├── model\
│   ├── Type1.py
│   ├── Type2.py
│   └── Type3.py
├── utils
│   ├── eval.py 
│   ├── network.py
│   ├── clustering_utils.py
│   ├── preprocessing.py
│   └── visualizer.py
├── constants.py
├── data_processing.py
├── clustering.py
├── main.py
└── tuning.py
```

## Implementation 

The implementation was tested with `Python 3.8.10` and the following packages:

- `numpy 1.22.4`
- `scipy 1.7.3`
- `pandas 1.5.3`
- `scikit-learn 1.2.0`
- `torch 1.13.1+cu116` 
- `tqdm 4.64.1`

In addition, `matplotlib 3.6.3` and `seaborn 0.12.2` were used for plots.

## Guideline

Note: This repository is under active cleaning and update.

### Data processing

```python data_processing.py```

### Clustering analysis

```python clustering.py```

### Model evaluation

```python main.py```


If you use any of the data or code, please cite [our paper](https://doi.org/10.1109/JBHI.2024.3368042)

```
V. Phan, K. Song, R. S. Silva, K. G. Silbernagel, J. R. Baxter and E. Halilaj, "Seven Things to Know About Exercise Classification With Inertial Sensing Wearables," in IEEE Journal of Biomedical and Health Informatics, vol. 28, no. 6, pp. 3411-3421, June 2024, doi: 10.1109/JBHI.2024.3368042.
```





