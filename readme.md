# Transformer Mixture Models
## Description
This project is an attempt at using mixture models approach with transformer models that draws inspiration from [Gaussian Mixture Model](https://brilliant.org/wiki/gaussian-mixture-model/) (GMM) algorithm (not 100% exact), [Stacking Ensemble](https://arxiv.org/abs/2104.02395) algorithm, and [Inception module](https://arxiv.org/abs/1409.4842). The core idea is to use multiple transformer models stacked together (side by side) and force them to focus on different features of the input, by simply concatenating their outputs. The concatenated output is either passed to a shared classifier layer or treated as the final output. The models don't communicate with each other, but are intended to act as one large model (Mixture).

The overall aim of such an approach is to scale the model without increasing model depth (number of layers), increasing the dimensions of the intermediate feed-forward layers, or using techniques such as [Mixture of Experts](https://arxiv.org/abs/2407.06204) (MoE), and their variants. Stacking is not performed in the layer of any of the models like with most implementations but rather as an ensemble of neural network models, because why not.

## Theory (How it works)
A Gaussian Mixture Model (GMM) is a probabilistic model that assumes data points are generated from a mixture of a finite number of Gaussian distributions, each with unknown parameters such as component weights, means, and variances (or covariances). By combining multiple simple (unimodal) Gaussian components and using the [Expectation–maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) algorithm, GMMs can perform soft clustering and estimate complex functions for density estimation. The core idea is that while a single Gaussian distribution cannot capture complex data patterns, a mixture of them provides greater flexibility and expressiveness, enabling the modeling of more intricate distributions.

<p align="center">
  <img alt="Gaussian Mixture Model for Density estimation example" src="assets/Gaussian%20Mixture%20Model.jpg" />
</p>

Ensemble learning is a machine learning technique that aggregates two or more models/learners in order to improve overall performance. One of the techniques to do this is called Stacking. Stacking involves using multiple learners that employ different algorithms such as KNN, Decision trees, etc, called base learners. They are all trained on the same dataset at first. After training, they are made to make predictions on an unseen dataset, held out during training, which are then aggregated and passed to a new model/learner called the meta learner, that's trained to make the final prediction, a process called meta-learning. Stacking typically yields better performance than any single one of the trained models.

<p align="center">
  <img alt="Stacking ensemble block diagram" src="assets/Stacking%20Ensemble.jpg" />
</p>

An inception module was a technique used for Computer Vision tasks (utilizing [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)) to increase the depth and width of a network while keeping the computational budget constant i.e keep the number of operations constant but adding more layers. In order to improve the performance of a model one had to increase the model size by either increasing the depth (adding more layers) or increasing width (number of "units" in each layer). The downsides of such an approach was that it made the model more prone to overfitting, where dataset size is small and it also dramatically increased use of computational resources. To address this, the inception module was introduced where the inputs dimension of the input are reduced and processed by stacked layers before finally being concatenated alongside their channels dimension for the next layer. The main benefits of doing this was it allowed for the increase of number of layers without a dramatic rise in computational complexity. This is mostly due to the dimension reduction in the module.

<p align="center">
  <img alt="Inception module block diagram" src="assets/Inception_module.jpg" />
</p>

Neural Networks are considered [universal function approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem), meaning they can approximate any continuous function given sufficient capacity. When provided with an input, such as a vector or a matrix (e.g images), a neural network transforms it through a series of operations to produce an output suitable for a specific task such as multi-class classification.

These transformations involve matrix multiplications, bias additions and non-linear activations, all of which are combined to form layers in model. The intermediate outputs between layers, often call feature vectors, can be visualized in abstract feature space, where each dimension represents a learned features/characteristics from the data.

The expressive power of a neural network depends on it's architecture, particularly its depth (number of layers) and width (size of intermediate vectors, controlled by weight matrices). While larger models are generally more capable, they also introduce greater computational complexity and are more prone to overfitting, especially when training data is limited

This project aims to combine the above mentioned concepts (GMM, Stacking Ensemble, and Inception Module) in such a manner that allows a neural network model, in this case a Transformer model to scale without needing to increase the depth or width. The core idea being to stack multiple models (side by side), mixture of models. The outputs of each model, which are usually reduced dimension-wise, are simply concatenated together and either treated as the final output or passed to a shared layer for final prediction. The concatenation operation ensures that each model's output focuses on different attributes of the data, and thereby specializing on a given feature/characterics. This is similar to how [Sparse Mixture of Experts](https://arxiv.org/abs/1701.06538) (MoE) work, where they utilize a mixture of experts that specialize on certain data / features by only allowing one expert to be active for a given data. This usually leads to them needing to employ complicated mechanisms such as router networks and auxiliary losses to ensure certain experts are not left underutilized (load balancing). My implementation does not have this issue, as no load balancing is needed. 

<p align="center">
  <img alt="Custom Transformer mixture model diagram" src="assets/Mixture%20Model.jpg" />
</p>

## Implementation
The project includes the following scripts and code:
- Script to generate Vocabulary dictionary.
- Script to generate subword dataset using Vocabulary.
- Script to generate train/test dataset.
- Script to generate testing loss .csv file using model checkpoints.
- Script to graph testing loss dataset from testing loss .csv files.
- Code to train model.

## Requirements
- Anaconda (Optional)
- Python 3

## Installing.
1. (Optional) Install [Anaconda](https://docs.anaconda.com/) on your machine.
    - Create anaconda environment:
    ```
    conda create --name <env_name> python=3.12
    ```
    - To activate anaconda environment, if not already activated:
    ```
    conda activate <env_name>
    ```
2. (If not installed) Install [Pytorch 2.5](https://pytorch.org/get-started/locally/) based on your hardware requirements and preferences.
3. Install Python depencies:
    ```
    pip install -r requirements.txt
    ```

## Dataset Creation (Needs to be run once)
Dataset used was [Plain Text Wikipedia 2020-11](https://www.kaggle.com/datasets/ltcmdrdata/plain-text-wikipedia-202011). Once downloaded, the json file was converted to raw plain-text files and all the Non-Latin characters like Greek symbols was removed (simplified the dataset). Script used for both operations are not provided!

### Generating vocabulary.
To generate vocabulary `.json` file, run the following (Needs to be generated first before everything else):
```
python scripts/generate_vocabulary.py <args>
```

### Generating subword dataset.
To generate subword dataset you need a `.csv` file listing all raw `.txt` files, run the following to generate a `.json` dataset file.
```
python scripts/generate_subword_dataset.py <args>
```

### Generating train/test dataset.
To generate the train/test dataset to be used for training model, run the following:
```
python scripts/generate_training_dataset.py <args>
```

## Training model.
To train the model, one first needs a config.json file (Example can be found in **training_results/Misc/example_config.json**), then run the following:
```
python train_mixture_models.py <args>
```

## Results
Models were trained on a single **RTX 3080** 10GB GPU. (For better quality charts, script to chart losses from `.csv` files and other miscellaneous files go to **training_results/** folder)

<p align="center">
  <img alt="All loss charts" src="assets/all_losses.png" />
</p>

## Interpretation
Two types of mixtures were tested:
- Mixture of models
- Mixture of blocks

### Mixture of Models
Here the concatenated models output were treated as class probabilities that were split amongst the model i.e each model focused on a group of class predictions, they were trained with a mixture of 1 and 3 models.

Observation: No improvement was observed.

Conclusion: Doesn't work at all.

### Mixture of Blocks
Here multiple layers forming the transformer model (Self-Attention, Feed Forward, Residual, and Norm layers) are grouped into what i call "block"s. Multiple "block"s can be sequentially stacked to increase model depth. Each "block" grouping contains their own Embedding and Positional Encoding layer, no sharing. The outputs of each "block" groupings are reduced vectors, determined beforehand by dividing the desired vector dimension with the number of mixtures of "blocks". A concatenation operation is then performed on all the vector outputs from each "block" groupings before being passed to a shared classifier layer.

Depth the model (number of "block" grouping) used was between 1 and 2, and number of mixtures of blocks used were 1, 2, 3, 5, 8, and 16 with some discrepancies in training steps.

Observation: As the number of mixtures kept increasing, for all depth sizes tested, the model seemed to perform better (lower testing loss). Some notable dimishing returns was observed when going from 8 to 16 mixtures, which should be expected as other factors like model depth became the limiting factor. It should be noted that, in some cases, shallow models (depth=1) with higher number of mixtures outperformed deeper models (depth=2) with no mixtures i.e baseline models. 

Conclusion: Seems to work well, up to a point. Needs to be tested at scale to see if trend holds. This architecture may benefit a lot from distributed computing. 

## Additional Relevant Links.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Ensemble deep learning: A review](https://arxiv.org/abs/2104.02395)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
