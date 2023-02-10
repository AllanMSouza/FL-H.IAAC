#!/bin/bash
#git checkout FedPer2
#git fetch
#git pull

echo "Executing joint plot..."
python3 analysis/joint_analysis.py



hostname=$(hostname)
if [[ $hostname == "claudio-Predator-PH315-52" ]]
then
  echo ""
else
  echo "Realizar commit"
  git config user.name "Cláudio"
  git config user.email claudiogs.capanema@gmail.com
  git add ./analysis/analysis/output
  git commit -m "Resultado da análise"
  git push origin FedPer2
fi