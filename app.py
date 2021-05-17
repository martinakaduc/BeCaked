from flask import Flask, request, jsonify, url_for, send_from_directory
from flask import render_template
from markupsafe import escape
from apscheduler.schedulers.background import BackgroundScheduler
import os
from datetime import datetime, timedelta
import argparse
import pickle
import time
import numpy as np
from waitress import serve

app = Flask(__name__)
sched = BackgroundScheduler(daemon=True)

@app.route("/hello", methods=["GET"])
@app.route('/hello/<name>', methods=["GET"])
def hello(name=None):
    return render_template('hello.html', name=escape(name))

@app.route("/", methods=["GET"])
def home():
    return render_template('home.html',
                            name='home',
                            request=request,
                            countries=countries,
                            data_countries_current=data_countries_current,
                            world_series=world_series,
                            world_series_predict=world_series_predict,
                            current_day=current_day)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_data', help='Wheather run prediction.', type=bool, default=False)
    parser.add_argument('--data_folder', help='Where to store website data.', type=str, default="./web_data")
    parser.add_argument('--cuda', help='Enable cuda', type=int, default=0)
    args = parser.parse_args()

    if args.cuda == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    init(args.init_data, args.data_folder)

    sched.add_job(update_data, 'interval', hours=6)
    sched.start()

    port = int(os.environ.get("PORT", 5000))
    serve(app, host='0.0.0.0', port=port)
    # app.run(debug=True, host='0.0.0.0', port=port)
