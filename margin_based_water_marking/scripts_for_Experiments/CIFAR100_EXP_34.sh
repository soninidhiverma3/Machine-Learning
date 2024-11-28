python train.py      --dataset cifar100    --train-type margin  --model-type res34
python train.py      --dataset cifar100    --train-type base    --model-type res34
python distill.py    --dataset cifar100    --train-type margin  --model-type res34
python distill.py    --dataset cifar100    --train-type base    --model-type res34
python extraction.py  --dataset cifar100    --train-type margin   --model-type res34
python extraction.py  --dataset cifar100    --train-type base   --model-type res34
