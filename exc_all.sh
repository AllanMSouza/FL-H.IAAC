#!/bin/bash
git checkout FedPer2
git fetch
git pull

#echo "Executing experiment 1..."
#python3 exec_simulation.py --experiment_id=1 > execution_log/experiment_1.txt 2>&1

echo "Executing experiment 2..."
python3 exec_simulation.py --experiment_id=2 > execution_log/experiment_2.txt 2>&1

#echo "Executing experiment 3..."
#python3 exec_simulation.py --experiment_id=3 > execution_log/experiment_3.txt 2>&1
#
#echo "Executing experiment 4..."
#python3 exec_simulation.py --experiment_id=4 > execution_log/experiment_4.txt 2>&1
#
#echo "Executing experiment 5..."
#python3 exec_simulation.py --experiment_id=5 > execution_log/experiment_5.txt 2>&1

hostname=$(hostname)
if [[ $hostname == "claudio-Predator-PH315-52" ]]
then
  echo ""
else
  echo "Realizar commit"
  git config user.name "Cláudio"
  git config user.email claudiogs.capanema@gmail.com
  git add ./analysis/output
  git commit -m "Resultado de experimento"
  git push origin FedPer2
fi