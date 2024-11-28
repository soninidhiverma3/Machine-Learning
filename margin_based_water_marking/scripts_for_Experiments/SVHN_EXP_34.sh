python train.py      --dataset svhn    --train-type margin  --model-type res34

python train.py      --dataset svhn    --train-type base    --model-type res34

python distill.py    --dataset svhn    --train-type margin  --model-type res34

python distill.py    --dataset svhn    --train-type base    --model-type res34

python extraction.py  --dataset svhn    --train-type margin   --model-type res34

python extraction.py  --dataset svhn    --train-type base   --model-type res34

