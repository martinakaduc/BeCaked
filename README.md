# BeCaked: An Explainable Artificial Intelligence Model For COVID-19 Forecasting

## How to use

### Install required packages

```
pip install -r requirements.txt
```

### Validation

```
python evaluation.py [-h] [--level LEVEL] [--day_lag DAY_LAG] [--step STEP]
                     [--start_date START_DATE] [--end_date END_DATE]
                     [--run_comparison RUN_COMPARISON]
                     [--plot_prediction PLOT_PREDICTION]
                     [--plot_param PLOT_PARAM] [--image_folder IMAGE_FOLDER]
                     [--cuda CUDA]
```

```
Optional arguments:
  -h, --help            show this help message and exit
  --level LEVEL         0: world; 1: countries; 2: both
  --day_lag DAY_LAG     The number of day lag.
  --step STEP           The number of forecasting step.
  --start_date START_DATE
                        The start day from which to make prediction.
  --end_date END_DATE   The end date of prediction.
  --run_comparison RUN_COMPARISON
                        Wheather to compare model.
  --plot_prediction PLOT_PREDICTION
                        Wheather to plot prediction.
  --plot_param PLOT_PARAM
                        Wheather to plot parameters.
  --image_folder IMAGE_FOLDER
                        Where to save plotted pictures.
  --cuda CUDA           Enable cuda
```

### Start web app

Edit [.env](.env) file for following arguments
```
Optional arguments:
  INIT_DATA             Whether run prediction.
  DATA_DIR              Where to store website data.
  CUDA_VISIBLE_DEVICES  GPU device
  PORT                  Deployment port
```

Then run
```
bash run_web.sh
```

## Step-by-step reproduce experiments

### Environment

In our experiments, we use a computer with below configurations.
```
Processor Intel(R) Core(TM) i7-4940MX CPU @ 3.10GHz 3.30 GHz
RAM       32.0 GB
GPU       NVIDIA Quadro K2100M 2GB

Python 3.7.5
```

### Step 1: Choosing suitable day lab number
You should run the following commands.
```
python evaluation.py --level 0 --day_lag 7 --step 31 --start_date 161 --end_date 192 --run_comparison 1 --cuda 1 | grep ... > results/world_7_step_31.txt
python evaluation.py --level 0 --day_lag 10 --step 31 --start_date 161 --end_date 192 --run_comparison 1 --cuda 1 | grep ... > results/world_10_step_31.txt
python evaluation.py --level 0 --day_lag 14 --step 31 --start_date 161 --end_date 192 --run_comparison 1 --cuda 1 | grep ... > results/world_14_step_31.txt
```

After that, you will have 3 result files located in **results** folder. Then, you can use those files to compare performance and choose the best suitable day lag number.

### Step 2: Compare our model with others at world level
The command to do this task is the same as the one provided in *Part 1*.
```
python evaluation.py --level 0 --day_lag 10 --step 31 --start_date 161 --end_date 192 --run_comparison 1 --cuda 1 | grep ... > results/world_10_step_31.txt
```

### Step 3: Compare our model with others at country level
To reproduce this task, you need to run the evaluation 3 times with different *step* number. The commands for this task are provided below.
```
python evaluation.py --level 1 --day_lag 10 --step 1 --start_date 161 --end_date 192 --run_comparison 1 --cuda 1 | grep ... > results/countries_10_step_1.txt
python evaluation.py --level 1 --day_lag 10 --step 7 --start_date 161 --end_date 192 --run_comparison 1 --cuda 1 | grep ... > results/countries_10_step_7.txt
python evaluation.py --level 1 --day_lag 10 --step 15 --start_date 161 --end_date 192 --run_comparison 1 --cuda 1 | grep ... > results/countries_10_step_15.txt
```

### Step 4: Get plotted figures of forecasting results at world level
You should run the below command. After successfully running this command, the figures will be saved in **images/world** folder.
```
python evaluation.py --level 0 --day_lag 10 --start_date 161 --end_date 192 --plot_prediction 1 --plot_param 1 --cuda 1
```

### Step 4: Get plotted figures of forecasting results at country level
You should run the below command. After successfully running this command, the figure swill be saved in **images/[COUNTRY NAME]** folder.
```
python evaluation.py --level 1 --day_lag 10 --start_date 161 --end_date 223 --plot_prediction 1 --plot_param 1 --cuda 1
```
