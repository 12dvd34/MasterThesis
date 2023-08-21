# Overview

## Dataset

CIFAR10.py provides access to the CIFAR-10 dataset. It offers
two functions `get_train_dataloader` and `get_test_dataloader`
which return a Pytorch dataloader. It also holds a `classes`
list of CIFAR-10 classes which can be used to output the
classification in a user-friendly manner.

CombinedDataloader.py concatenates the two given dataloaders
and provides one dataloader that can be used to train and
test on both dataloaders easily.

## Model

Resnet18.py acts as a wrapper for an (untrained) Resnet18 model,
which can be accessed through the `model` field. It also offers
`train` and `test` functions to train and evaluate the model
as well as `save` and `load` to manage persistence.

Resnet18Lightning.py provides a Pytorch Lightning module that can
be used for training and testing. It also provides the `save` and
`load` functions. To introduce an adversarial attack to training,
set the `attack` field to an attack from Attacks.py. The optimizer
used during training can be overwritten with an optimizer returned
from Optimizers.py using the `optimizer` field. The `loss` field
can be overwritten too.

FeatureExtractor can be used to collect the output of a layer.
First, initialise it by passing the respective layer, then
run training or inference on the model. The FeatureExtractor
will collect activations on each forward pass. The number
of outputs collected can be limited with the `max_outputs`
argument. Access the outputs by calling `get_features`.

NoiseInjector will, given a layer, introduce noise to the
output of the layer during forward passes. The noise added
to the output is drawn  from a normal distribution with a 
mean at 0 and scale relative to the scale of the layers 
output distribution. The default of 1, meaning that the 
scale of the noise will equal the outputs scale, can be 
adjusted using the `scale` parameter.

## Adversarial Attacks

Attacks.py provides the following adversarial attacks:
- FGSM
- IFGSM
- PGD
- DeepFool (requires batch size > 1)

To use them, instantiate the attack and apply it to the
data.

It also offers the `AttackDataloader`, which, while not
being an actual Pytorch dataloader, acts like one for all
purposes within this project. It will apply the provided
`attack` on a given number of samples from the
underlying dataloader and can be used with the `train` and
`test` functions of Resnet18.

## Analysis

Analysis.py provides methods to analyse the model. 
- `get_weights` will return all weights in a single, flattened tensor. 
By default, it will access all layers. A list of layers can be passed to limit the scope, 
e.g. `[model.fc]`.
- `get_activations` works the same way, but will return activations instead.
- `get_distribution` is a wrapper around scipy's `stats.norm.fit` that takes a tensor, 
like the one's returned by the two previous functions, and returns location and scale 
of a normal distribution.
- `analyse_distributions` takes a list of layers and collects
weight and activation distributions from them using the
previously mentioned functions.
- `get_parameter_diffs` takes two tensors of weights or 
activations and computes the differences.
- `analyse_parameter_diffs` will compute the difference as
defined by PyTorch L1Loss of weights and activations
between the given models for the given layers. The models
should have the same structure (e.g. Resnet18) and the
layers should be the same for both models.
- `dim_reduce` takes a tensor of parameters as returned by
FeatureExtractor and applies T-SNE dimension reduction.
- `get_feature_clusters` takes a list of features 
(as returned by either the FeatureExtractor or `dim_reduce`) and
a list of labels and computes clusters for the features
according to their labels. It returns a list of dictionaries
containing the cluster centroids and mean distance of datapoints
to the centroid for each label.

## Plotting

Plotting can be done by using Plot.py. Use any of the following
functions and then call `show` to display the plot or
`save` to save it to a file.
- `plot_histogram` takes either weights or activations; 
to make comparisons between histograms easier, the x-axis
and bins can be synchronised by passing the combined data
of all histograms with the `comparison` argument
- `plot_distribution` takes distribution parameters
- `plot_features` takes features as returned by FeatureExtractor
or `Analysis.dim_reduce` and optionally a list of labels to
color the datapoints according to their labels
- `plot_feature_clusters` takes clusters as returned by
`Analysis.get_feature_clusters` and plots them as circles

Please note that only histogram/distribution and
features/clusters can be plotted on one figure.

## Displaying Images

Images, original and perturbed, can be displayed using Image.py.
Instantiate and Image by passing the tensor. This only works
with batches of size 1. Then, call `show` to display the
image on screen. Call `show_diff` and pass another image
to show an image visualising the differences between the
two images.

## Utilities

Utils.py contains some helper functions that might be useful.
- `save_tensor_to_csv` does exactly what is says
- `get_labels` extracts the labels from a dataset given
a dataloader and returns them as a list
- `normalize_tensor` scales all values in a tensor into
the range between 0 and 1

## Noise

Noise can be injected into a network by instantiating
the NoiseInjector from Resnet18Lightning.py and passing
a layer. During forward passes, noise will be added to
the output of that layer. Please refer to the NoiseInjector
paragraph of the Model section for details.