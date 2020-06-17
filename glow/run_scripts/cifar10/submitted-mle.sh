DR=data

K=32
J=0
WG=0
MLE=1
GP=0
AMS=0
db=0
SAVE=${ROOT1}/flow-gan
LR=5e-4
bs=64
LT=0
h=512
for exp in 0 1; do
AMS=0
GP=0
db=0
perm=reverse
case "$exp" in
0)
    c=additive
    eps=0
;; 
1)
    c=affine
    eps=0.01
;;  
esac

cmd="train.py  \
    --gan  \
    --dataset cifar10  \
    --L 3  \
    --K ${K} \
    --hidden_channels $h  \
    --batch_size $bs \
    --n_init_batches 10 \
    --disc_lr ${LR}  \
    --flow_permutation ${perm}   \
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
    --output_dir ${SAVE}/cifar10-mle-big/${c}-${exp}-resumed \
    --eval_every 1000 \
    --optim_name adamax \
    --svd_every 100000000000 \
    --no_warm_up 0 \
    --lr ${LR} \
    --max_grad_clip  ${GP}  \
    --no_conv_actnorm 0 \
    --actnorm_max_scale ${AMS} \
    --logittransform $LT \
    --epochs 500 \
    --db ${db} \
    "

python $cmd

done
