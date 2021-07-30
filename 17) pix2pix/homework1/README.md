# Generating Simpsons Scenes from sketches using a pix2pix model

In this Notebook, we will use the [Simpsons dataset](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset/) to train a pix2pix model to generate Simpsons scenes from simple drawings. 

The pix2pix allows to build generative models conditioned on spatial inputs (i.e. images)

paper:
* [pix2pix](https://arxiv.org/abs/1611.07004)


# pix2pix.ipynb파일안의 내용을 완성할것

## Model

The pix2pix model uses

* a U-Net model as generative model $G$. This model takes an image $y$ (or other 2D input) as input and produces another image $x$ (or other 2D input)
* a convolutional dscriminator model $D$. This model takes both $x$ and $y$ as input and tries to guess if $x \sim p_x$ or $x \sim p_G$

The model is trained using the GAN loss function and an additional $L_1$ loss on the output of the generative model $G$. Both losses are balanced using an additional hyperparameter $lambda$ (set to 100 in the original paper), the loss is:

$L = L_{GAN} + \lambda * L_1$


## You must to implement this two class and visualization result using `vis.py`: 

### Generator $G$
`from network import Generator`

### Discriminator $D$
`from network import Discriminator`