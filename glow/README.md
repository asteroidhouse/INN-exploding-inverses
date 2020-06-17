# Understanding and Mitigating Exploding Inverses in Invertible Neural Networks

Based on `https://github.com/y0ast/Glow-PyTorch`


## Setup and run

The code has minimal dependencies. You need python 3.6+ and up to date versions of:

```
pytorch (tested on 1.1.0)
torchvision
pytorch-ignite
tqdm
```

Set environment variable `ROOT1` to your favorite location.


## Reproduce the Additive/Affine MLE models

**Train**
```
bash run_scripts/cifar10/submitted-mle.sh
```

**Stability Analysis**
```
bash run_scripts/cifar10/submitted-run-analyze.sh
```

## Reproduce FlowGAN
```
bash run_scripts/cifar10/submitted-flowgan.sh
```
