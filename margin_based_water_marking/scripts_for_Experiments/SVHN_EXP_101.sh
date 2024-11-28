python train.py      --dataset svhn    --train-type margin  --model-type res101
python train.py      --dataset svhn    --train-type base    --model-type res101
python distill.py    --dataset svhn    --train-type margin  --model-type res101
python distill.py    --dataset svhn    --train-type base    --model-type res101
python extraction.py  --dataset svhn    --train-type margin   --model-type res101
python extraction.py  --dataset svhn    --train-type base   --model-type res101
