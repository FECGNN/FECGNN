# Graph Neural Networks with Fractionally Exponentiated Convolutions

>

This repository contains code for the paper "Graph Neural Networks with Fractionally Exponentiated Convolutions".

Graph neural networks have become increasingly popular in recent years to learn node representations from graphs. However, most graph convolutional networks are not very deep due to the challenges caused by over-smoothing. Over-smoothing refers to the fact that the convolution operator causes all nodes to have very similar representations, as the number of layers increases. 
This is naturally caused by the neighborhood aggregation operator.
In this study, we first propose a fractional convolution operator to reduce the impact of over-smoothing, by adjusting the scaling factors in the correlation directions corresponding to the eigenvectors of the adjacency matrix. 
Second, we develop a graph structure optimization method by adaptively dropping nodes with high-level heterophily to further alleviate the over-smoothing.
Third, we design a graph neural network FECGNN that combines the fractional convolution operator and adaptive structure optimization to learn effective representations with deep layers.
We finally conduct an extensive experimental study to verify the advantages of our approach for deep graph convolutional networks.


## Installation
We ran our code on Ubuntu 22.04 Linux, using Python 3.10, PyTorch 1.11 and CUDA 11.3.


Use the following code to install python packages:


```sh
conda env create -f environment.yml
```


Unzip the ```data.zip```



## Running FECGNN
```sh
python src/run_model.py --model_selec v6 --dataset cora --epochs 2000 --lr 0.0005 --wd 0.002 --wd2 0.002 --dropout 0.6 --exp_frac 0.8 --num_layer 50 --norm_type gcn --fVer 202408 --num_mlp_layers 2 --mlp_hidden_dim 64 --hidden_fixed 64 --early_stop --patiences 400 --output_skip --no_frac_save --exp_func 2 --drop_redundant --p_drop_rate 0.25 --wnh_ver 2310 --cuda
```



