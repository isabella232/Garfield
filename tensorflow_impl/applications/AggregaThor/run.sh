#!/bin/bash

OAR_NODEFILE=nodes

mkdir -p config
rm config/*
uniq $OAR_NODEFILE | python3 config_generator.py

main=$( realpath ./trainer.py --relative-base $( git rev-parse --show-toplevel ) )
node_base=./Garfield

for filename in config/*; do
    IP=$(echo $( basename $filename ) | cut -c11- | tr -d '\n')
   
    cmd=". ./Garfield/tensorflow_impl/venv_tensorflow/bin/activate && python3 ${node_base}/${main} \
        --config /tmp/$( basename ${filename}) \
        --log True         \
        --max_iter 2000    \
        --batch_size 64    \
        --dataset cifar10  \
        --nbbyzwrks 3      \
        --model VGG"

    echo "Running ${cmd} on ${IP}"

    scp -i ~/.vagrant.d/insecure_private_key ${filename} vagrant@${IP}:/tmp/
    ssh -i ~/.vagrant.d/insecure_private_key vagrant@${IP} "${cmd}" &

    # oarsh $IP python3 Garfield_TF/applications/MSMW/trainer.py \
    #     --config Garfield_TF/applications/MSMW/$filename \
    #     --log True         \
    #     --max_iter 2000    \
    #     --batch_size 64    \
    #     --dataset cifar10  \
    #     --nbbyzwrks 3      \
    #     --model VGG &
done
