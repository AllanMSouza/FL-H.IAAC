#!/bin/bash
#git checkout FedPer2
#git fetch
#git pull

#echo "Executing experiment 1..."
#python3 exec_experiments.py --experiment_id=1 > execution_log/experiment_1.txt 2>&1

#echo "Executing experiment 2..."
#python3 exec_experiments.py --experiment_id=2 --type="torch" > execution_log/experiment_2.txt 2>&1

#echo "Executing experiment 3..."
#python3 exec_experiments.py --experiment_id=3 --type="torch" > execution_log/experiment_3.txt 2>&1
#
#echo "Executing experiment 4..."
#python3 exec_experiments.py --experiment_id=4 --type="torch" > execution_log/experiment_4.txt 2>&1
#
#echo "Executing experiment 5..."
#python3 exec_experiments.py --experiment_id=5 --type="torch" > execution_log/experiment_5.txt 2>&1

#echo "Executing experiment 6..."
#python3 exec_experiments.py --experiment_id=6 --type="torch" > execution_log/experiment_6.txt 2>&1
#
#echo "Executing experiment 7..."
#python3 exec_experiments.py --experiment_id=7 --type="torch" > execution_log/experiment_7.txt 2>&1
#
#echo "Executing experiment 8..."
#python3 exec_experiments.py --experiment_id=8 --type="torch" > execution_log/experiment_8.txt 2>&1

#echo "Executing experiment 10..."
#python3 exec_experiments.py --experiment_id=10 --type="torch" > execution_log/experiment_10.txt 2>&1

echo "Executing experiment 11..."
python3 exec_experiments.py --experiment_id=11 --type="torch" > execution_log/experiment_11.txt 2>&1

echo "Executing experiment 12..."
python3 exec_experiments.py --experiment_id=12 --type="torch" > execution_log/experiment_12.txt 2>&1

echo "Executing experiment 13..."
python3 exec_experiments.py --experiment_id=13 --type="torch" > execution_log/experiment_13.txt 2>&1

#hostname=$(hostname)
#if [[ $hostname == "claudio-Predator-PH315-52" ]]
#then
#  echo ""
#else
#  echo "Realizar commit"
#  git config user.name "Cláudio"
#  git config user.email claudiogs.capanema@gmail.com
#  git add ./analysis/output
#  git commit -m "Resultado de experimento"
#  git push origin FedPer2
#fi