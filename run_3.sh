#!/bin/bash
export TIMES=63

echo "QUAN 5"
for i in {1..40}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 5" --img_note _QUAN_5_$i
    echo $i
    echo '\n'
  done

echo "QUAN 6"
for i in {1..40}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "QUAN 6" --img_note _QUAN_6_$i
    echo '\n'
  done

echo "CU CHI"
for i in {1..40}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "CU CHI" --img_note _CU_CHI_$i
    echo $i
    echo '\n'
  done

echo "TAN BINH"
for i in {1..40}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "TAN BINH" --img_note _TAN_BINH_$i
    echo '\n'
  done

echo "TAN PHU"
for i in {1..40}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "TAN PHU" --img_note _TAN_PHU_$i
    echo '\n'
  done
