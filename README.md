# Reliable-KL-estimation

This repository is the official PyTorch implementation of [Reliable Estimation of KL Divergence using a Discriminator in Reproducing Kernel Hilbert Space](https://papers.nips.cc/paper/2021/file/54a367d629152b720749e187b3eaa11b-Paper.pdf). 
<!---
>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials
--->
## Requirements

To install requirements:
The code has been tested on Python 3.7, PyTorch 1.6.0. The requirements are minimal : numpy, pandas, matplotlib. You could also run the following:
```setup
pip install -r requirements.txt
```
<!---
[comment]: #  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...
--->
## Training, Evaluation, Plotting
### Mutual Information Estimation
To estimate mutual information, run the following three commands:

```train
python kl_main.py
```

```train
python kl_lip_features.py --lambd 0
```
```train
python kl_lip_features.py --lambd 1e-05
```
```train
python kl_lip_features.py --lambd 0.001
```
```train
python kl_lip_features.py --lambd 0.1
```
```train
python kl_lip_features.py --lambd 1
```
```train
python plot_method_array_short.py
```
<!---
>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
--->