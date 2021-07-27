#!/bin/bash
export TIMES=61

echo "BINH CHANH"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "BINH CHANH" --img_note _BINH_CHANH_$i
    echo '\n'
  done

echo "BINH TAN"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "BINH TAN" --img_note _BINH_TAN_$i
    echo '\n'
  done
