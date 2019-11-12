## Setup
1. Create a conda environment using `requirements.txt` using the command provided below.

    ```conda create --name img-classifcation --file requirements.txt```
2. `requirements.txt` already contains Pytorch 1.3 which is supposed to run on a GPU. If you plan to run this code on CPU, install Pytorch using this command. Make sure to be inside the conda environment `img-classification` using installing it.
    ```conda install pytorch torchvision cpuonly -c pytorch```

## Contributing

Write your model architecture inside `models` folder. `models.softmax.py` has been provided for reference.

Scripts inside `scripts` can be used to train the models. `scripts/run_softmax.sh` has been provided for reference.
