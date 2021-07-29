#!/bin/bash
export TIMES=63
export LOOP=40

echo "QUAN 1"
for i in {1..$LOOP}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 1" --img_note _QUAN_1_$i
    echo $i
    echo '\n'
  done

echo "QUAN 10"
for i in {1..$LOOP}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 10" --img_note _QUAN_10_$i
    echo $i
    echo '\n'
  done

echo "QUAN 7"
for i in {1..$LOOP}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 7" --img_note _QUAN_7_$i
    echo '\n'
  done

echo "QUAN 4"
for i in {1..$LOOP}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 4" --img_note _QUAN_4_$i
    echo '\n'
  done

echo "QUAN 8"
for i in {1..$LOOP}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 8" --img_note _QUAN_8_$i
    echo '\n'
  done

echo "QUAN 3"
for i in {1..$LOOP}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 3" --img_note _QUAN_3_$i
    echo $i
    echo '\n'
  done
