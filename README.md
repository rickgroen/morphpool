# MorphPool: Efficient Non-linear Pooling & Unpooling in CNNs
This is the repository for our BMVC 2022 paper titled **MorphPool: Efficient Non-linear Pooling & Unpooling in CNNs**. \
**The paper link is coming soon.**


## Dependencies
A [requirements](requirements.txt) file is available to retrieve all dependencies. Create a new python environment and install using:
```shell
pip install -r requirements.txt
``` 


## CUDA operations for pooling & unpooling
This paper makes available CUDA implementations for MorphPool, since morphological operations have not yet been implemented in CUDA for neural networks.
The operations can be compiled using
```commandline
cd morphology
sh setup.sh
```
This requires you to have installed all dependencies, and have the CUDA toolkit installed on your system.
After compilation, some unit tests are run to check whether compilation was successful.


## Training and testing
You can train any model using *train.py* by specifying your down and up-sampling operation.
See [options.py](options/options.py) for the available operations. \
In general, it should look like:
```commandline
python train.py --pool_method param_pool --pool_ks 3 --unpool_method param_morph_unpool --unpool_ks 5
```
At the end of training, the weights are saved automatically to [output](output/).
From these model files, you can test performance on the test sets using *test.py*.
This is done by invoking *test.py* with the same arguments you used during training. \
For example, the model that was trained using the command above, can be tested by:
```commandline
python test.py --pool_method param_pool --pool_ks 3 --unpool_method param_morph_unpool --unpool_ks 5
```

## Datasets
This repository allows you to run our code on three semantic segmentation datasets: NYU, SUN-RGBD and 2D3D-S.
Check out [loaders](loaders/README.md) to download and setup the datasets for your system. \
After you have done so, you can train a model on a specific using the --dataset argument for [ nyu | sun | 2d3ds ]. \
For example
```commandline
python train.py --dataset sun
```
Note that you have to change the value of **DATA_FOLDER** on line 2 in [config.py](config.py) to the path to the folder you keep your datasets.

Finally, it is good to note that we use the simple network UpDownNet as introduced in the paper.
It was meant to isolate changes in performance due to sampling, not reach the SOTA in semantic segmentation.
Consequently, it has relatively few parameters, is not pre-trained on ImageNet, has no specific semantic modules, and does not have to be trained for many iterations.
Feel free to apply the (morphological) up and down-sampling in more complex architectures.

## Citation
If this work was useful for your research, please consider citing:
```
[Coming soon]
```
