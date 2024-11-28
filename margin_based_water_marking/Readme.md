# Abstract

As Machine Learning as a Service (MLaaS) platforms become prevalent, deep neural network (DNN) watermarking techniques are gaining increasing attention, which enables one to verify the ownership of a target DNN model in a black-box scenario. Unfortunately, previous watermarking methods are vulnerable to functionality stealing attacks, thus allowing an adversary to falsely claim the ownership of a DNN model stolen from its original owner. In this work, we propose a novel margin-based DNN watermarking approach that is robust to the functionality stealing attacks based on model extraction and distillation. Specifically, during training, our method maximizes the margins of watermarked samples by using projected gradient ascent on them so that their predicted labels cannot change without compromising the accuracy of the model that the attacker tries to steal. We validate our method on multiple benchmarks and show that our watermarking method successfully defends against model extraction attacks, outperforming relevant baselines.

The project aims to develop a watermarking method for machine learning models to protect intellectual property, especially in environments like Machine Learning as a Service (MLaaS) platforms. The specific objectives are:To create a watermarking approach that can reliably verify model ownership in black-box settings.To make the watermark robust against functionality-stealing attacks, such as model extraction and distillation, which can undermine traditional watermarking methods.


## All Packages are mentioned in requirements.txt

## Creating Environment
1. conda create -n my_env  python=3.10
2. conda activate my_env
3. pip install -r requirements.txt
4. pip install wandb
5. wandb login    ## FOR WANDB LOGIN  API key   744fdd72c58a6a31f2854e210631419a8ee30b05





## Training a model on a Dataset using a method  eg dataset=shvn , model=res101 , attack_type=margin
"python train.py   --dataset svhn    --train-type margin   --model-type res101"

## Distilation a model 
"python distill.py    --dataset svhn    --train-type margin  --model-type res101"

## Extract Watermark
"python extraction.py  --dataset svhn    --train-type margin   --model-type res101"


## How to Do Experiments on datasets   use folder Scripts_for_Experiments
let us do experiment on shvn dataset with resnet101 using method margin,
fro that I create a script  
1. "SHVN_EXP_101.sh"
2. run it using cli "cd scripts/bash SHVN_EXP_101.sh" in terminal
3. All results are stored in experiments folder


## Results Visualization
1. All results like logs ,  images , Plots  are stored and can be seen on wandb under project_name "Watermark-ML-Project"


## Link for Report
https://docs.google.com/document/d/1doK-dTmw2m8CUXXzE2w6c-fH1SNphQgGhCwin7ZvZ40/edit?usp=sharing
