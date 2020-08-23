# Understanding and Mitigating Exploding Inverses in Invertible Neural Networks

This repository contains the code used for the paper [Understanding and mitigating exploding inverses in invertible neural networks](http://arxiv.org/abs/2006.09347).

This code is based on [jhjacobsen/fully-invertible-revnet](https://github.com/jhjacobsen/fully-invertible-revnet), [y0ast/Glow-PyTorch](https://github.com/y0ast/Glow-PyTorch), [plucas14/pytorch-glow](https://github.com/pclucas14/pytorch-glow), and [rtqichen/residual-flows](https://github.com/rtqichen/residual-flows).


## Requirements

* Python 3.7.x
* PyTorch 1.1.0


## Setup

First, create a conda environment with the necessary packages:
```
conda create -n inn-env python=3.7
source activate inn-env
conda install pytorch=1.1.0 cuda80 -c pytorch
conda install torchvision -c pytorch
pip install -r requirements.txt
```


## Experiments

### Out-of-Distribution Evaluation for Pre-Trained Models

The following commands should be run from inside the `ood-pretrained` folder.

First, download the OOD datasets:
```
./download_ood_data.sh
```

#### Evaluate Pre-Trained Glow on OOD Data

Download the pre-trained Glow model used in the [y0ast/Glow-PyTorch](https://github.com/y0ast/Glow-PyTorch) repository:
```
./download_pretrained_glow.sh
```

Then, run the OOD evaluation:
```
python glow_ood_eval.py
```

#### Evaluate a Pre-Trained Residual Flow on OOD Data

The following commands must be run within the `ood-pretrained/residual-flow` folder.

First, download the pre-trained Residal Flow model from the repository [rtqichen/residual-flows](https://github.com/rtqichen/residual-flows):
```
./download_pretrained_resflow.sh
```

Then, run the OOD evaluation:
```
python resflow_ood_eval.py
```


### Invertibility Attack

Details on how to set up and run the invertibility attack are provided in the README inside the `inv-attacks-and-2D-flows` folder.


### Flow Training/Stability Analysis and FlowGAN

For details on how to train and analyze stability properties of flows, as well as how to train a FlowGAN, see the README inside the `glow` folder.


### Toy 2D Regression

To train regularized and unregularized Glow models on the toy 2D regression task, run the following commands from inside the `toy-2d-regression` folder:

```
python toy_2d_regression.py
python toy_2d_regression.py --nf_coeff=1e-6
```


### Classification

To train an INN classifier, run any of the following commands from inside the `classification` folder:

#### Training an unregularized affine model

To train an unregularized model that becomes numerically non-invertible, you can run:

**Unregularized Affine Glow with 1x1 Convolutions**
```
python train_inn_classifier.py \
    --coupling=affine \
    --permutation=conv \
    --zero_init \
    --use_prior \
    --use_actnorm \
    --no_inverse_svd \
    --save_dir=saves/affine_conv
```

**Unregularized Affine Glow with Shuffle Permutations**
```
python train_inn_classifier.py \
    --coupling=affine \
    --permutation=shuffle \
    --zero_init \
    --use_prior \
    --use_actnorm \
    --no_inverse_svd \
    --save_dir=saves/affine_shuffle
```

#### Training with normalizing flow (NF) regularization

**Affine Glow with 1x1 Convolutions, NF coefficient 1e-5**
```
python train_inn_classifier.py \
    --coupling=affine \
    --permutation=conv \
    --zero_init \
    --use_prior \
    --use_actnorm \
    --no_inverse_svd \
    --nf_coeff=1e-5 \
    --save_dir=saves/affine_conv
```

**Affine Glow with Shuffle Permutations, NF coefficient 1e-4**
```
python train_inn_classifier.py \
    --coupling=affine \
    --permutation=shuffle \
    --zero_init \
    --use_prior \
    --use_actnorm \
    --no_inverse_svd \
    --nf_coeff=1e-5 \
    --save_dir=saves/affine_shuffle
```


#### Training with memory-saving gradients using finite differences (FD) regularization

**Affine Glow with 1x1 Convolutions, FD coefficient 1e-4**
```
python train_inn_classifier.py \
    --coupling=affine \
    --permutation=conv \
    --zero_init \
    --use_prior \
    --use_actnorm \
    --no_inverse_svd \
    --fd_coeff=1e-4 \
    --fd_inv_coeff=1e-4 \
    --regularize_every=10 \
    --mem_saving \
    --save_dir=saves/affine_conv
```

**Affine Glow with Shuffle Permutations, FD coefficient 1e-4**
```
python train_inn_classifier.py \
    --coupling=affine \
    --permutation=shuffle \
    --zero_init \
    --use_prior \
    --use_actnorm \
    --no_inverse_svd \
    --fd_coeff=1e-4 \
    --fd_inv_coeff=1e-4 \
    --regularize_every=10 \
    --mem_saving \
    --save_dir=saves/affine_shuffle
```


## Citation

If you find this repository useful, please cite:

* `Jens Behrmann*, Paul Vicol*, Kuan-Chieh Wang*, Roger Grosse, Jörn-Henrik Jacobsen. Understanding and mitigating exploding inverses in invertible neural networks, 2020.`

```
@article{innexploding2020,
  title={Understanding and mitigating exploding inverses in invertible neural networks},
  author={Jens Behrmann and Paul Vicol and Kuan-Chieh Wang and Roger Grosse and Jörn-Henrik Jacobsen},
  journal={arXiv preprint arXiv:2006.09347},
  year={2020}
}
```
