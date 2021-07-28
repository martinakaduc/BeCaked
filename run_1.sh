#!/bin/bash
export TIMES=61

echo "THU DUC"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "THU DUC" --img_note _THU_DUC_$i
    echo $i
    echo '\n'
  done

echo "GO VAP"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "GO VAP" --img_note _GO_VAP_$i
    echo $i
    echo '\n'
  done

echo "BINH THANH"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "BINH THANH" --img_note _BINH_THANH_$i
    echo '\n'
  done

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

echo "CAN GIO"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "CAN GIO" --img_note _CAN_GIO_$i
    echo $i
    echo '\n'
  done

echo "CU CHI"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "CU CHI" --img_note _CU_CHI_$i
    echo $i
    echo '\n'
  done

echo "HOC MON"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "HOC MON" --img_note _HOC_MON_$i
    echo '\n'
  done

echo "NHA BE"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "NHA BE" --img_note _NHA_BE_$i
    echo '\n'
  done

echo "PHU NHUAN"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "PHU NHUAN" --img_note _PHU_NHUAN_$i
    echo '\n'
  done

echo "TAN BINH"
for i in {1..30}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $TIMES --cuda 1 --ward "TAN BINH" --img_note _TAN_BINH_$i
    echo '\n'
  done
