#!/bin/bash
export TIMES=61

echo "QUAN 1"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 1" --img_note _QUAN_1_$i
    echo $i
    echo '\n'
  done

echo "QUAN 10"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 10" --img_note _QUAN_10_$i
    echo $i
    echo '\n'
  done
