# CS3612FinalProject

## Image Classification on MNIST 

Before training, the scripts will first download datasets to a `dataset` folder.
For image classification, train with 
```bash
bash train_mnist.sh
```
And evaluate the model with best checkpoint with this script:
```bash
bash visualize_mnist.sh
```
This script will generate visualization results in `images` folder and corresponding information on test set. `mnist_feature_map%d_%s.png` represents the feature map on `%d` layer with `%s` visualization method. For example, `mnist_feature_map3_tsne.png` is the result of feature map of the third layer with `t-SNE` visualization method. `mnist-train-loss-acc.png` represents the change of loss and accuracy on training set.

## Image Generation on FashionMNIST
Train with the following script:
```bash
bash train_fashion_mnish.sh
```
And evaluate the model with best checkpoint with this script:
```bash
bash visualize_vae.sh
```
This script will generate visualization results in `images` folder. `vae-image1/2.png` are randomly generated images with two arbitrary Gaussian noise. `vae-%.2f-merge.png` is the image generated with interpolation.

## Sentiment Analysis on SST-2 
Train with the following script:
```bash
bash train_sst2.sh
```
And evaluate the model with best checkpoint with this script:
```bash
bash visualize_sst2.sh
```
This script will generate visualization results in `images` folder.
`sst2_feature_map%d_%s.png` represents the feature map on `%d` layer with `%s` visualization method. `sst2-%s-loss-acc.png` represents the loss and accuracy change figure either in training set or in validation set.