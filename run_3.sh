#!/bin/bash
export TIMES=61

echo "QUAN 4"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 4" --img_note _QUAN_4_$i
    echo '\n'
  done

echo "QUAN 8"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 8" --img_note _QUAN_8_$i
    echo '\n'
  done
