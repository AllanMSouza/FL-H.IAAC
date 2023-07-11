#!/bin/bash
git checkout FedPredict-v2.1
git fetch
git pull

hostname=$(hostname)
if [[ $hostname == "claudio-Predator-PH315-52" ]]
then
  echo ""
else
  echo "Realizar commit"
  git config user.name "Cláudio"
  git config user.email claudiogs.capanema@gmail.com
  git add ./logs
  git commit -m "Resultado de experimento"
  git push origin FedPredict-v2.1
fi