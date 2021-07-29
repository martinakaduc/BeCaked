#!/bin/bash
export TIMES=63

echo "QUAN 11"
for i in {1..40}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 11" --img_note _QUAN_11_$i
    echo '\n'
  done

echo "QUAN 12"
for i in {1..40}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 12" --img_note _QUAN_12_$i
    echo '\n'
  done

echo "HOC MON"
for i in {1..40}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "HOC MON" --img_note _HOC_MON_$i
    echo '\n'
  done

echo "NHA BE"
for i in {1..40}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "NHA BE" --img_note _NHA_BE_$i
    echo '\n'
  done

echo "PHU NHUAN"
for i in {1..40}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "PHU NHUAN" --img_note _PHU_NHUAN_$i
    echo '\n'
  done
