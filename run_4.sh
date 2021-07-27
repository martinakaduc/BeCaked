#!/bin/bash
export TIMES=61

echo "QUAN 7"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 7" --img_note _QUAN_7_$i
    echo '\n'
  done

echo "BINH THANH"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "BINH THANH" --img_note _BINH_THANH_$i
    echo '\n'
  done
