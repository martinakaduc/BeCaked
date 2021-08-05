#!/bin/bash
export TIMES=71
export END_TRAIN=66

echo "THU DUC"
for i in {1..50}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $END_TRAIN --cuda 1 --ward "THU DUC" --img_note THU_DUC_$i
    echo $i
    echo '\n'
  done

echo "GO VAP"
for i in {1..50}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $END_TRAIN --cuda 1 --ward "GO VAP" --img_note GO_VAP_$i
    echo $i
    echo '\n'
  done

echo "BINH THANH"
for i in {1..50}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $END_TRAIN --cuda 1 --ward "BINH THANH" --img_note BINH_THANH_$i
    echo '\n'
  done

echo "BINH CHANH"
for i in {1..50}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $END_TRAIN --cuda 1 --ward "BINH CHANH" --img_note BINH_CHANH_$i
    echo '\n'
  done

echo "BINH TAN"
for i in {1..50}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $END_TRAIN --cuda 1 --ward "BINH TAN" --img_note BINH_TAN_$i
    echo '\n'
  done

echo "CAN GIO"
for i in {1..50}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $END_TRAIN --cuda 1 --ward "CAN GIO" --img_note CAN_GIO_$i
    echo $i
    echo '\n'
  done

echo "CU CHI"
for i in {1..50}
  do
    python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $END_TRAIN --cuda 1 --ward "CU CHI" --img_note CU_CHI_$i
    echo $i
    echo '\n'
  done

echo "HCM"
for i in {1..50}
  do
  python evaluation.py  --level 0 --day_lag 10 --start_date 43 --end_date $TIMES --plot_prediction 1 --infer_date 7 --start_train_date 0 --end_train_date $END_TRAIN --cuda 1 --ward "HCM" --img_note HCM_$i
  echo $i
  echo '\n'
done
