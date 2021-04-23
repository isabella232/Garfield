pwd=`pwd`
while read p; do
	ssh -i ~/.vagrant.d/insecure_private_key vagrant@${p%:*} "pkill -f trainer.py" < /dev/tty
done < "workers"
while read p; do
        ssh -i ~/.vagrant.d/insecure_private_key vagrant@${p%:*} "pkill -f trainer.py" < /dev/tty
done < "servers"
