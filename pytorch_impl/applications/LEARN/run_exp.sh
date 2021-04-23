#!/bin/bash

nodes_names=nodes
num_nodes=10000			#The upper bound on the number of nodes in any deployment
f=0
iter=100000			#100000
#dataset='cifar10'
dataset='mnist'
#model=resnet50		#resnet18
model=convnet		#resnet18
optimizer='sgd'
batch=125			#250
loss='cross-entropy'
lr=0.2				#0.1
gar='average'			#'average'
master=''
nnodes=0
while read p; do
    	nnodes=$((nnodes+1))
	if [ $nnodes -eq  1 ]
  	then
     		master=$p
  	fi
        if [ $num_nodes -eq $nnodes ]
        then
                break
        fi
done < $nodes_names

pwd=`pwd`
main=$( realpath ./trainer.py --relative-base $( git rev-parse --show-toplevel ) )
node_base=./Garfield
#CUDAHOSTCXX=/usr/bin/gcc-5
common=". ./Garfield/pytorch_impl/venv_pytorch/bin/activate && python3 ${node_base}/${main} --master $master --num_iter $iter --dataset $dataset --model $model --batch $batch --loss $loss"
common="$common --optimizer $optimizer --opt_args '{\"lr\":\"$lr\",\"momentum\":\"0.9\",\"weight_decay\":\"0.0005\"}' --num_nodes $nnodes --f $f --gar $gar"
i=0
while read p; do
        cmd="$common --rank $i"
        ssh -i ~/.vagrant.d/insecure_private_key vagrant@$p "$cmd" < /dev/tty &
        echo "running $cmd on $p"
        i=$((i+1))
        if [ $i -eq $nnodes ]
        then
                break
        fi
done < $nodes_names
