DR=data
K=32
J=0
WG=0
MLE=1
GP=0
AMS=0
db=0
SAVE=${ROOT1}/flow-gan
for LR in 5e-4; do
for bs in 64; do
for LT in 0; do
for h in 512; do
for exp in 0 1; do
for iter in 0 1000 2000 3000 4000 5000 10000 20000 50000 100000; do
AMS=0
GP=0
db=0
case "$exp" in
0)
    c=additive
    eps=0
;; 
1)
    c=affine
    eps=0.01
;;  
2)
    c=additive
    eps=0
    AMS=5
;;
3)
    c=affine
    eps=0.01
    AMS=5
;;
esac

output_dir=${SAVE}/cifar10-mle-big/${c}-${exp}

cmd="analyze.py  \
    --dataset cifar10  \
    --dataroot ${DR} \
    --glow_path ${output_dir}/ckpt_${iter}.pt"



python $cmd


done
done
done
done
done
done
