#!/bin/bash
echo "Executing FedAVG and FedSGD with no Selection mechanism ..."
python3 exec_simulation.py -a None --non-iid True

#echo "Executing FedAVG and FedSGD with POC ..."
#python3 exec_simulation.py -a POC --non-iid True

#echo "Executing FedAVG and FedSGD with FedLTA ..."
#python3 exec_simulation.py -a FedLTA --non-iid True


