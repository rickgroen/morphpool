## Data Preparation
This repository allows you to run our code on three datasets: NYU, SUN-RGBD and 2D3D-S.
Here's how to download and setup the datasets on your system.

Note that you have to change the value of **DATA_FOLDER** on line 2 in [config.py](../config.py) to the path to the folder you keep your datasets.
For me, my data lives in */home/data*, with folders for each separate dataset.

There are a number of data preparation scripts in the [loaders/utils/](utils) folder.
You may have to move these scripts to locations at which you unpack and store your data.

### NYU
Download the NYUv2 dataset from the project website.
After that you can unpack and store the data in your **[DATA_FOLDER]**. 
Before training, you have to run [utils/prepare_nyu.py](utils/prepare_nyu.py) to set up the directory structure used for this code.
It also uses a colorization scheme to fill in the depth maps.
References to the respective papers and repositories are given in the code.

### SUN-RGBD
The dataset can be downloaded using the shell script [utils/download_sunrgbd.sh](utils/download_sunrgbd.sh), which downloads it to the folder you execute the script in.
Move the data and unpack it using [utils/unpack_sunrgbd.sh](utils/unpack_sunrgbd.sh).

### 2D3D-S
Download the raw 2D3D-S from the project website.
After that you can unpack and store the data in your **[DATA_FOLDER]**. 
Before training, you have to run [utils/prepare_2d3ds.py](utils/prepare_2d3ds.py) to set up the directory structure used for this code.
It also uses a colorization scheme to fill in the depth maps.
References to the respective papers and repositories are given in the code.