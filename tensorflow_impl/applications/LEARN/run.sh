#!/bin/bash

OAR_NODEFILE=nodes

mkdir -p config
rm -r config/*
uniq $OAR_NODEFILE | python3 config_generator.py

main=$( realpath ./trainer.py --relative-base $( git rev-parse --show-toplevel ) )
node_base=./Garfield

for filename in config/*; do
    IP=$(echo $( basename $filename ) | tr -d '\n')
    
    cmd=". ./Garfield/tensorflow_impl/venv_tensorflow/bin/activate && python3 ${node_base}/${main} \
        --config_w /tmp/TF_CONFIG_W \
	    --config_ps /tmp/TF_CONFIG_PS \
       	--log True         \
        --max_iter 2001     \
        --batch_size 64    \
        --dataset cifar10  \
    	--nbbyzwrks 3      \
        --model Cifarnet"

    echo "Running ${cmd} on ${IP}"

    scp -i ~/.vagrant.d/insecure_private_key ${filename}/* vagrant@${IP}:/tmp/
    ssh -i ~/.vagrant.d/insecure_private_key vagrant@${IP} "${cmd}" &

    # oarsh $IP python3 Garfield_TF/applications/LEARN/trainer.py \
    #     --config_w Garfield_TF/applications/LEARN/config/$IP/TF_CONFIG_W \
	#     --config_ps Garfield_TF/applications/LEARN/config/$IP/TF_CONFIG_PS \
    #    	--log True         \
    #     --max_iter 2001     \
    #     --batch_size 64    \
    #     --dataset cifar10  \
    # 	--nbbyzwrks 3      \
    #     --model Cifarnet &
done
