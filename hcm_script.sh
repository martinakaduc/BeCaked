#!/bin/bash
export TIMES=66
export END_TRAIN=61

echo "HCM"
for i in {1..50}
  do
  python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $END_TRAIN --cuda 1 --ward "HCM" --img_note HCM_$i
  echo $i
  echo '\n'
done
