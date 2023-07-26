#!/bin/bash
#git checkout FedPredict-v2.1
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

#echo "Executing experiment 7..."
#python3 exec_experiments.py --experiment_id=7 --type="torch" > execution_log/experiment_7.txt 2>&1
#
#echo "Executing experiment 8..."
#python3 exec_experiments.py --experiment_id=8 --type="torch" > execution_log/experiment_8.txt 2>&1

#echo "Executing experiment 10..."
#python3 exec_experiments.py --experiment_id=10 --type="torch" > execution_log/experiment_10.txt 2>&1
#
#echo "Executing experiment 11..."
#python3 exec_experiments.py --experiment_id=11 --type="torch" > execution_log/experiment_11.txt 2>&1
#
#echo "Executing experiment 12..."
#python3 exec_experiments.py --experiment_id=12 --type="torch" > execution_log/experiment_12.txt 2>&1
#
#echo "Executing experiment 13..."
#python3 exec_experiments.py --experiment_id=13 --type="torch" > execution_log/experiment_13.txt 2>&1
#
echo "Executing experiment 14..."
python3 exec_experiments.py --experiment_id=14 --type="torch" > execution_log/experiment_14.txt 2>&1

#echo "Executing experiment 15..."
#python3 exec_experiments.py --experiment_id=15 --type="torch" > execution_log/experiment_15.txt 2>&1
#
#echo "Executing experiment 16..."
#python3 exec_experiments.py --experiment_id=16 --type="torch" > execution_log/experiment_16.txt 2>&1
#
#echo "Executing experiment 17..."
#python3 exec_experiments.py --experiment_id=17 --type="torch" > execution_log/experiment_17.txt 2>&1

#echo "Executing experiment 18..."
#python3 exec_experiments.py --experiment_id=18 --type="torch" > execution_log/experiment_18.txt 2>&1

#echo "Executing experiment 19..."
#python3 exec_experiments.py --experiment_id=19 --type="torch" > execution_log/experiment_19.txt 2>&1

#echo "Executing experiment 20..."
#python3 exec_experiments.py --experiment_id=20 --type="torch" > execution_log/experiment_20.txt 2>&1
#
#echo "Executing experiment 21..."
#python3 exec_experiments.py --experiment_id=21 --type="torch" > execution_log/experiment_21.txt 2>&1
#
#echo "Executing experiment 22..."
#python3 exec_experiments.py --experiment_id=22 --type="torch" > execution_log/experiment_22.txt 2>&1
#
#echo "Executing experiment 23..."
#python3 exec_experiments.py --experiment_id=23 --type="torch" > execution_log/experiment_23.txt 2>&1
#
#echo "Executing experiment 24..."
#python3 exec_experiments.py --experiment_id=24 --type="torch" > execution_log/experiment_24.txt 2>&1
#
#echo "Executing experiment 25..."
#python3 exec_experiments.py --experiment_id=25 --type="torch" > execution_log/experiment_25.txt 2>&1

echo "Executing experiment 26..."
python3 exec_experiments.py --experiment_id=26 --type="torch" > execution_log/experiment_26.txt 2>&1

#echo "Executing experiment 27..."
#python3 exec_experiments.py --experiment_id=27 --type="torch" > execution_log/experiment_27.txt 2>&1
#
#echo "Executing experiment 28..."
#python3 exec_experiments.py --experiment_id=28 --type="torch" > execution_log/experiment_28.txt 2>&1
#
#echo "Executing experiment 29..."
#python3 exec_experiments.py --experiment_id=29 --type="torch" > execution_log/experiment_29.txt 2>&1

echo "Executing experiment 30..."
python3 exec_experiments.py --experiment_id=30 --type="torch" > execution_log/experiment_30.txt 2>&1

#echo "Executing experiment 31..."
#python3 exec_experiments.py --experiment_id=31 --type="torch" > execution_log/experiment_31.txt 2>&1

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