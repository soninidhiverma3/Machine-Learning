python train.py      --dataset cifar100    --train-type margin  --model-type res101
python train.py      --dataset cifar100    --train-type base    --model-type res101
python distill.py    --dataset cifar100    --train-type margin  --model-type res101
python distill.py    --dataset cifar100    --train-type base    --model-type res101
python extraction.py  --dataset cifar100    --train-type margin   --model-type res101
python extraction.py  --dataset cifar100    --train-type base   --model-type res101
