## Image classification using supervised and unsupervised methods

## Description
The method description and the results are provided [here](https://rishabhjain.xyz/ml-class-project/).

## Setup

1.Create a conda environment using `requirements.txt` using the command provided below.

```conda create --name img-classifcation --file requirements.txt```

2.`requirements.txt` already contains Pytorch 1.3 which is supposed to run on a GPU. If you plan to run this code on CPU, install Pytorch using this command. Make sure to be inside the conda environment `img-classification` while installing it.

    ```conda install pytorch torchvision cpuonly -c pytorch```

## How to run a model?
After the requirements are installed, open a terminal to run the model as -

```bash scripts/run_<model_name>.sh```

## Contributing

Write your model architecture inside `models` folder. `models.softmax.py` has been provided for reference.

Scripts inside `scripts` can be used to train the models. `scripts/run_softmax.sh` has been provided for reference.
