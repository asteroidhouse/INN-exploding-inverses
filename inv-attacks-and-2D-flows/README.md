# Understanding and Mitigating Exploding Inverses in Invertible Neural Networks

This folder contains:

- The invertibility attack on CelebA64 (including model training)
- 2D toy experiments on checkerboard data

Built using the "Residual Flows for Invertible Generative Modeling" (https://papers.nips.cc/paper/9183-residual-flows-for-invertible-generative-modeling) repository from: https://github.com/rtqichen/residual-flows


## 2D Checkerboard Experiments: Training and Evaluation

To train a residual flow use `train_toy_resflow.py` with the default setting and to train an affine model use `train_toy_resflow.py`.

Note: to modify the scaling in the affine model, one currently has to do it manually in '\lib\layers\coupling.py' by commenting out lines 65 and 73.

Evaluation (plotting) is done in `analyze_results.py`.


## Preprocessing for CelebA64:

Based on CelebAHQ 256x256 data (downsampling to 64x64 done later in data loader)
```
# Download Glow's preprocessed dataset.
wget https://storage.googleapis.com/glow-demo/data/celeba-tfr.tar
tar -C data/celebahq -xvf celeb-tfr.tar
python extract_celeba_from_tfrecords.py
```


## Density Estimation on CelebA64: Training

CelebA64 Residual Flow:
```
python train_img_adapted.py --data celeba64_5bit --imagesize 64 --nbits 5 --actnorm True --block resblock --act swish --n-exact-terms 8 --fc-end True --factor-out False --squeeze-first True --nblocks 16-16-16-16 --save experiments/celebA64resflow
```

CelebA64 Affine model:
````
python train_img_adapted.py --data celeba64_5bit --imagesize 64 --nbits 5 --actnorm True --block coupling --act elu --update-freq 5 --n-exact-terms 8 --fc-end True --factor-out False --squeeze-first True --nblocks 16-16-16-16 --optimizer adamax --save experiments/celeba64affine
````

Note: to modify the scaling in the affine model, one currently has to do it manually in `\lib\layers\coupling.py` by outcommenting in line 65 and 73.


## Density Estimation on CelebA64: Evaluation via Invertibility Attack

Use `attack_celebA.py` for the invertibility attack. Note: The hyperparameters need to be selected manually.
