from flask import Flask, request, jsonify, url_for, send_from_directory
from flask import render_template
from markupsafe import escape
import os
from datetime import datetime, timedelta
import argparse
import pickle
import time
import numpy as np
import json

from database import get_latest_data, get_daily_latest_statistics

app = Flask(__name__)

@app.route("/hello", methods=["GET"])
@app.route('/hello/<name>', methods=["GET"])
def hello(name=None):
    return render_template('hello.html', name=escape(name))

@app.route("/old-home", methods=["GET"])
def old_home():
    return render_template('old_home.html',
                            name='home',
                            request=request,
                            countries=countries,
                            data_countries_current=data_countries_current,
                            world_series=world_series,
                            world_series_predict=world_series_predict,
                            current_day=current_day)

@app.route("/", methods=["GET"])
@app.route("/<district>", methods=["GET"])
def home(district="hcm"):
    district = district.upper()
    district = district.replace('-',' ')
    
    backup_data_dir = os.environ.get("BACKUP_DATA_PATH", "./backup/")
    backup_data_path = os.path.join(backup_data_dir,district+'.json')
    backup_summary_path = os.environ.get("BACKUP_SUMMARY_PATH", "./backup/backup_summary.json")
    with open(backup_data_path) as json_file:
        data = json.load(json_file)
    with open(backup_summary_path) as json_file:
        summary = json.load(json_file)

    districts = ['BINH CHANH', 'BINH TAN', 'BINH THANH', 'CAN GIO', 'CU CHI', 'GO VAP', 'HCM', 'HOC MON', 'NHA BE', 'PHU NHUAN'] + [f'QUAN {i}' for i in [1, 3, 4, 5, 6, 7, 8, 10, 11, 12]] + ['TAN BINH', 'TAN PHU', 'THU DUC']
    districts.remove('HCM')
    districts.sort()
    districts.append('HCM')

    return render_template('home.html',
                            name = district,
                            today = data['_id'],
                            districts = districts,
                            summary = summary['data'],
                            data = data['data'],
                            num_cols = [3 + (district == 'HCM'),1 + (district == 'HCM'),1]
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

@app.route("/whitepaper", methods=["GET"])
def whitepaper():
    return render_template('whitepaper.html',
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
    init(run_init, data_dir)

    return app

if __name__ == "__main__":
    app = main()
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
