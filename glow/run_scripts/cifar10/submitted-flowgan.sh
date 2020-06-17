DR=data
AMS=0
K=8
J=0
WG=1
SAVE=${ROOT1}/flow-gan
for LR in 5e-5; do
for DLR in 1e-4 5e-5; do
for exp in 2 3; do
for A in mine; do
for MLE in 1e-3 1e-4; do
case "$exp" in
0)
    c=additive
    eps=0
;; 
1)
    c=gaffine
    eps=0
;;  
2)
    c=affine
    eps=0.01
;;
3)
    c=affine
    eps=0.01
    AMS=5
;;

esac


python train.py  \
    --fresh  \
    --gan  \
    --dataset cifar10  \
    --L 3  \
    --K ${K} \
    --hidden_channels 128  \
    --batch_size 64 \
    --n_init_batches 0 \
    --disc_lr ${DLR}  \
    --flow_permutation reverse   \
    --flow_coupling ${c}  \
    --affine_max_scale 5 \
    --affine_scale_eps 2 \
    --affine_eps ${eps} \
    --weight_gan ${WG} \
    --weight_prior ${MLE} \
    --weight_logdet ${MLE} \
    --jac_reg_lambda ${J} \
    --flowgan 1 \
    --dataroot ${DR} \
    --output_dir ${SAVE}/cifar10/reg-mle1/${MLE}-${A}-${exp}-${LR}-${DLR} \
    --eval_every 1000 \
    --optim_name adam \
    --svd_every 100000000000 \
    --no_warm_up 1 \
    --lr ${LR} \
    --actnorm_max_scale ${AMS} \
    --max_grad_clip  0  \
    --no_conv_actnorm 0 \
    --disc_arch $A \
    --no_learn_top 


done
done
done
done
done
