from flask import Flask, request, jsonify, url_for, send_from_directory
from flask import render_template
from flask.helpers import make_response
from markupsafe import escape
import os
from datetime import datetime, timedelta
import argparse
import pickle
import time
import numpy as np
import json

from database import get_latest_data, get_daily_latest_statistics, check
from utils import *

app = Flask(__name__)

@app.route('/reload-db', methods=["POST"])
def reload():
    try:
        token = request.json['token']
        if check(token):
            os.system('python3 database.py')
            return "done"
        else:
            return "invalid token"
    except:
        return "invalid token"

@app.route("/hello", methods=["GET"])
@app.route('/hello/<name>', methods=["GET"])
def hello(name=None):
    return render_template('hello.html', name=escape(name))

@app.route("/old-home", methods=["GET"])
def old_home():
    return render_template('old_home.html',
                            name='old-home',
                            request=request,
                            countries=countries,
                            data_countries_current=data_countries_current,
                            world_series=world_series,
                            world_series_predict=world_series_predict,
                            current_day=current_day)

# @app.route("/", methods=["GET"])
# @app.route("/<district>", methods=["GET"])
# def home(district="hcm"):
#     district = district.upper()
#     district = district.replace('-',' ')
    
#     backup_data_dir = os.environ.get("BACKUP_DATA_PATH", "./backup/")
#     backup_data_path = os.path.join(backup_data_dir,district+'.json')
#     backup_summary_path = os.environ.get("BACKUP_SUMMARY_PATH", "./backup/backup_summary.json")
#     with open(backup_data_path) as json_file:
#         data = json.load(json_file)
#     with open(backup_summary_path) as json_file:
#         summary = json.load(json_file)

#     districts = ['BINH CHANH', 'BINH TAN', 'BINH THANH', 'CAN GIO', 'CU CHI', 'GO VAP', 'HCM', 'HOC MON', 'NHA BE', 'PHU NHUAN'] + [f'QUAN {i}' for i in [1, 3, 4, 5, 6, 7, 8, 10, 11, 12]] + ['TAN BINH', 'TAN PHU', 'THU DUC']
#     districts.remove('HCM')
#     districts.sort()
#     districts.append('HCM')

#     return render_template('home.html',
#                             name = district,
#                             today = data['_id'],
#                             districts = districts,
#                             summary = summary['data'],
#                             data = data['data'],
#                             num_cols = [6,3,1]
#                             )

@app.route('/__load_data')
def load():
    state = request.args['state']
    scenario = request.args['scenario']
    district = ' '.join(request.args['district'].split('_')).upper()
    backup_data_dir = os.environ.get("BACKUP_DATA_PATH", "./backup/")
    backup_data_path = os.path.join(backup_data_dir,district+'.json')
    content = json.load(open(backup_data_path))
    dates = content['data'].pop('dates')
    if scenario == 'best':  
        data = content['data'][state]['BestCase']
        scenario = 'Best Scenario'
    elif scenario == 'normal': 
        data = content['data'][state]['NormalCase']
        scenario = 'Normal Scenario'
    elif scenario == 'worst': 
        data = content['data'][state]['WorstCase']
        scenario = 'Worst Scenario'
    actual_data = content['data'][state]['real']

    if state.upper() in ['R','D','V']:
        temp = np.array(data)
        temp = temp[1:] - temp[:-1]
        temp[temp < 0] = 0
        data[1:] = temp.tolist()
        # data[0] -= actual_data[-1]

        temp = np.array(actual_data)
        temp = temp[1:] - temp[:-1]
        temp[temp < 0] = 0
        actual_data[1:] = temp.tolist()
    
    data = data[1:]
    actual_data = actual_data[1:]
    dates = dates[1:]

    num_real = len(actual_data)
    actual_data.extend([0] * (len(data) - len(actual_data)))
    mask = 1 - np.array(actual_data).astype(bool).astype(int)
    data = (np.array(data) * mask).tolist()
    return json.dumps({'scenario': scenario, 'dates': dates, 'actual': actual_data, 'data': data})

@app.route("/", methods=["GET"])
def home():
    districts = ['BINH CHANH', 'BINH TAN', 'BINH THANH', 'CAN GIO', 'CU CHI', 'GO VAP', 'HCM', 'HOC MON', 'NHA BE', 'PHU NHUAN'] + [f'QUAN {i}' for i in [1, 3, 4, 5, 6, 7, 8, 10, 11, 12]] + ['TAN BINH', 'TAN PHU', 'THU DUC']
    backup_data_dir = os.environ.get("BACKUP_DATA_PATH", "./backup/")
    total, month, week = {},{},{}
    for district in districts:
        backup_data_path = os.path.join(backup_data_dir,district+'.json')
        with open(backup_data_path) as json_file:
            data = json.load(json_file)
            raw = data['data']
            total[district] = get_daily_data(raw)
            month[district] = get_nth_last_data(total[district],28)
            week[district] = get_nth_last_data(total[district],7)
            today = data['_id']
            
            #get yesterdate
            today = datetime.strptime(today,'%m.%d') - timedelta(1)
            today = today.replace(year=2021,hour=18)
            today = today.strftime('%b %d, %Y %H:%M')

    backup_summary_path = os.environ.get("BACKUP_SUMMARY_PATH", "./backup/backup_summary.json")
    with open(backup_summary_path) as json_file:
        summary = json.load(json_file)['data']

    districts.sort(key=lambda x:summary[x]['I']['Total'],reverse=True)
    districts_v = sorted(districts,key=lambda x:summary[x]['V']['Total'],reverse=True)
    centre = {'BINH CHANH': [10.7080941, 106.59928437], 'BINH THANH': [10.81236736, 106.7106739], 'BINH TAN': [10.76849142, 106.59185577], 'CU CHI': [11.03145621, 106.52149395], 'CAN GIO': [10.508327, 106.8635], 'GO VAP': [10.83800042, 106.66737587], 'HOC MON': [10.88712536, 106.60970238], 'NHA BE': [10.64094082, 106.72282509], 'PHU NHUAN': [10.79897935, 106.67950392], 'QUAN 10': [10.77079414, 106.66866466], 'QUAN 11': [10.76583559, 106.64520678], 'QUAN 12': [10.86093358, 106.65810283], 'QUAN 1': [10.78343482, 106.69548463], 'QUAN 3': [10.78522737, 106.67608034], 'QUAN 4': [10.75829395, 106.70296998], 'QUAN 5': [10.75301895, 106.66929241], 'QUAN 6': [10.74503421, 106.63090987], 'QUAN 7': [10.73376706, 106.72356962], 'QUAN 8': [10.72540308, 106.6435674], 'TAN BINH': [10.81229902, 106.65917519], 'TAN PHU': [10.78907531, 106.62464563], 'THU DUC': [10.82217494, 106.77374533]}
    population = { "HCM": 8926959, "QUAN 1": 139485, "QUAN 3": 189258, "QUAN 4": 173970, "QUAN 5": 157920, "QUAN 6": 232077, "QUAN 7": 356380, "QUAN 8": 422151, "QUAN 10": 233223, "QUAN 11": 208680, "QUAN 12": 618365, "BINH CHANH": 702972, "BINH TAN": 781417, "BINH THANH": 495955, "CAN GIO": 69326, "CU CHI": 457275, "GO VAP": 671252, "HOC MON": 539227, "NHA BE": 205329, "PHU NHUAN": 162148, "TAN BINH": 470393, "TAN PHU": 478786, "THU DUC": 1161370}

    return render_template('hcm.html',
                            name = 'HCM',
                            today = today,
                            districts = districts,
                            districts_v = districts_v,
                            centre = centre,
                            population = population,
                            total = total,
                            month = month,
                            week = week,
                            summary = summary
                            )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if ('day-start' in request.form and 'day-end' in request.form):
        day_start = datetime.strptime(request.form['day-start'], "%Y-%m-%d")
        day_end = datetime.strptime(request.form['day-end'], "%Y-%m-%d")
        number_of_days = (day_end - day_start).days
        start_index = (day_start - datetime.strptime("2020-01-22", "%Y-%m-%d")).days

        return render_template('predict.html',
                                name='predict',
                                data_series=np.array(data_series)[:,start_index:start_index+number_of_days].tolist(),
                                start_day=request.form['day-start'],
                                num_day=number_of_days,
                                date_series=date_series[start_index:start_index+number_of_days])

    return render_template('predict.html',
                            name='predict',
                            data_series=data_series,
                            start_day='2020-01-22',
                            num_day=len(data_series[0]),
                            date_series=date_series)

@app.route("/about", methods=["GET"])
def whitepaper():
    return render_template('about.html',
                            name='whitepaper')

@app.route("/donate", methods=["GET"])
def donate():
    return render_template('donate.html',
                            name='donate')

@app.route("/achievements", methods=["GET"])
def achievements():
    return render_template('achievements.html',
                            name='achievements')

@app.route("/acknowledgement", methods=["GET"])
def acknowledgement():
    return render_template('acknowledgement.html',
                            name='acknowledgement')

@app.route("/contact", methods=["GET"])
def contact():
    return render_template('contact.html',
                            name='contact')

@app.route("/googlede2ce4a4cee74360.html", methods=["GET"])
def googlede2ce4a4cee74360():
    return render_template('googlede2ce4a4cee74360.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico')

def load_predict_result(data_folder):
    global countries
    global world_series
    global world_series_predict
    global data_series
    global date_series
    global current_day
    global data_countries_current

    countries              = pickle.load(open(data_folder + "/countries.pkl", 'rb'))
    world_series           = pickle.load(open(data_folder + "/world_series.pkl", 'rb'))
    world_series_predict   = pickle.load(open(data_folder + "/world_series_predict.pkl", 'rb'))
    data_series            = pickle.load(open(data_folder + "/data_series.pkl", 'rb'))
    date_series            = pickle.load(open(data_folder + "/date_series.pkl", 'rb'))
    current_day            = pickle.load(open(data_folder + "/current_day.pkl", 'rb'))
    data_countries_current = pickle.load(open(data_folder + "/data_countries_current.pkl", 'rb'))

def init(run_predict=False, data_folder="./"):
    if not run_predict:
        load_predict_result(data_folder)
        return

    from data_utils import DataLoader
    from becaked import BeCakedModel
    data_loader = DataLoader()
    becaked_model = BeCakedModel()

    global countries
    countries = data_loader.get_countries()
    pickle.dump(countries, open(data_folder + "/countries.pkl", "wb"))

    global world_series # For home page
    world_series = [x.tolist()[-30:] for x in data_loader.get_data_world_series()]
    world_series.insert(1, [x[0] - x[1] - x[2] for x in zip(world_series[0], world_series[1], world_series[2])])
    pickle.dump(world_series, open(data_folder + "/world_series.pkl", "wb"))

    global world_series_predict
    world_series_predict = [[], [], [], []]

    for _ in range(30):
        world_series[0].append(None)
        world_series[1].append(None)
        world_series[2].append(None)
        world_series[3].append(None)

    global data_series
    world_series_temp = [x.tolist() for x in data_loader.get_data_world_series()]
    data_series = [[], [], [], []]

    for i in range(10):
        data_series[0].append(world_series_temp[0][i])
        data_series[1].append(world_series_temp[0][i] - world_series_temp[1][i] - world_series_temp[2][i])
        data_series[2].append(world_series_temp[1][i])
        data_series[3].append(world_series_temp[2][i])

    for i in range(len(world_series_temp[0]) - 10):
        res = becaked_model.predict([world_series_temp[0][i:i+10], world_series_temp[1][i:i+10], world_series_temp[2][i:i+10]])[0][-1]
        data_series[0].append(int(res[1] + res[2] + res[3]))
        data_series[1].append(int(res[1]))
        data_series[2].append(int(res[2]))
        data_series[3].append(int(res[3]))

    world_series_predict[0] = data_series[0][len(world_series_temp[0])-30:len(world_series_temp[0])+30]
    world_series_predict[1] = data_series[1][len(world_series_temp[0])-30:len(world_series_temp[0])+30]
    world_series_predict[2] = data_series[2][len(world_series_temp[0])-30:len(world_series_temp[0])+30]
    world_series_predict[3] = data_series[3][len(world_series_temp[0])-30:len(world_series_temp[0])+30]

    pickle.dump(data_series, open(data_folder + "/data_series.pkl", "wb"))
    pickle.dump(world_series_predict, open(data_folder + "/world_series_predict.pkl", "wb"))

    sd = datetime.strptime("2020-01-22", "%Y-%m-%d")
    global date_series
    date_series = []
    for i in range(len(data_series[0])):
        date_series.append((sd+timedelta(days=i)).strftime("%d / %m / %Y"))
    pickle.dump(date_series, open(data_folder + "/date_series.pkl", "wb"))

    global current_day
    current_day = data_loader.get_current_day()
    pickle.dump(current_day, open(data_folder + "/current_day.pkl", "wb"))

    global data_countries_current
    data_countries_current = data_loader.get_data_countries_current()
    data_countries_current.insert(1, {country:abs(data_countries_current[0][country] - data_countries_current[1][country] - data_countries_current[2][country])
                                    if country in data_countries_current[1] else abs(data_countries_current[0][country] - data_countries_current[2][country])
                                    for country in countries[0]})
    pickle.dump(data_countries_current, open(data_folder + "/data_countries_current.pkl", "wb"))

def update_data():
    print("Updating data!")
    os.system("rm -rf COVID-19/csse_covid_19_data/csse_covid_19_time_series")
    os.system("svn checkout --force https://github.com/CSSEGISandData/COVID-19/trunk/csse_covid_19_data/csse_covid_19_time_series COVID-19/csse_covid_19_data/csse_covid_19_time_series")
    time.sleep(30)
    init(run_predict=True)

def main():
    run_init = bool(os.environ.get("INIT_DATA", True))
    data_dir = str(os.environ.get("DATA_DIR", "./web_data"))
    # init(run_init, data_dir)

    return app

if __name__ == "__main__":
    app = main()
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
