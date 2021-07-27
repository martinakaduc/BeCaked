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
