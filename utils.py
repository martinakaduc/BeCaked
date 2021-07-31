import os
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.type_check import real
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.arima_model import ARIMA
from sklearn.svm import SVR
from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import matplotlib.ticker as plticker

def get_list_date(start=datetime(2021,6,26), length=50):
    temp = [(start + timedelta(days=i)).strftime("%d/%m") for i in range(length)]
    return temp

def update_data():
    os.system("cd COVID-19 && git pull origin master")

def get_predict_by_step(ml_model, data, start, current, end=None, day_lag=10, return_param=False):
    if end == None:
        end = data.shape[1]

    predict_data = data[:,:current]
    list_param_byu = []

    for day in range(predict_data.shape[1]-day_lag, end-day_lag):
        return_result = ml_model.predict([predict_data[0][day:day+day_lag],
                                          predict_data[1][day:day+day_lag],
                                          predict_data[2][day:day+day_lag],
                                          predict_data[3][day:day+day_lag],
                                          predict_data[4][day:day+day_lag]],
                                          return_param=return_param)
        if return_param:
            next_day, param_byu = return_result
            list_param_byu.append(param_byu[0])
        else:
            next_day = return_result

        next_day = next_day[0][-1]
        next_day[1] += next_day[2] + next_day[3]
        predict_data = np.append(predict_data, next_day[1:].reshape((-1, 1)), axis=1)

    if return_param and current >= start - day_lag:
        for k in range(current - start + day_lag):
            s = current-k-day_lag
            e = current-k
            return_result = ml_model.predict([data[0][s:e],
                                              data[1][s:e],
                                              data[2][s:e],
                                              data[3][s:e],
                                              data[4][s:e]],
                                              return_param=return_param)
            next_day, param_byu = return_result
            list_param_byu.insert(0, param_byu[0])

    return predict_data, np.array(list_param_byu)

def get_predict_result_1(ml_model, data, start, current, end=None, day_lag=10, return_param=False):
    """"Predict with real data"""
    predict_data = data[:,:start]
    list_param_byu = []
    if current + day_lag < end: raise Exception("Invalid end_date")
    for day in range(start-day_lag,end-day_lag):
        return_result = ml_model.predict([data[0][day:day+day_lag],
                                          data[1][day:day+day_lag],
                                          data[2][day:day+day_lag],
                                          data[3][day:day+day_lag],
                                          data[4][day:day+day_lag]],
                                          return_param=return_param)
        if return_param:
            next_day, param_byu = return_result
            list_param_byu.append(param_byu[0])
        else:
            next_day = return_result
        # print(next_day)
        # raise Exception("haiz")
        next_day = next_day[0][-1]
        predict_data = np.append(predict_data, next_day[1:].reshape((-1,1)), axis=1)

    if return_param and current >= start - day_lag:
        for k in range(current - start + day_lag):
            s = current-k-day_lag
            e = current-k
            return_result = ml_model.predict([data[0][s:e],
                                              data[1][s:e],
                                              data[2][s:e],
                                              data[3][s:e],
                                              data[4][s:e]],
                                              return_param=return_param)
            next_day, param_byu = return_result
            list_param_byu.insert(0, param_byu[0])

    return predict_data, np.array(list_param_byu)




def get_predict_result(ml_model, data, start, end=None, step=31, day_lag=10):
    predict_data = data[:,:start]
    if end == None: end = data.shape[1]

    confirmed = [0 for _ in range(end-start)]
    recovered = [0 for _ in range(end-start)]
    deceased = [0 for _ in range(end-start)]
    list_div = [0 for _ in range(end-start)]

    for ste in range(start, end-step+1):
        tem_predict = data[:,:ste]

        for day in range(ste, ste+step):
            next_day = ml_model.predict([tem_predict[0][day-day_lag:day], tem_predict[1][day-day_lag:day], tem_predict[2][day-day_lag:day]])[0][-1]
            next_day[1] += next_day[2] + next_day[3]
            tem_predict = np.append(tem_predict, next_day[1:].reshape((-1, 1)), axis=1)

            confirmed[day-start] += next_day[1]
            recovered[day-start] += next_day[2]
            deceased[day-start] += next_day[3]
            list_div[day-start] += 1

    confirmed = np.array(confirmed) / np.array(list_div)
    recovered = np.array(recovered) / np.array(list_div)
    deceased = np.array(deceased) / np.array(list_div)

    return np.append(predict_data, [confirmed, recovered, deceased], axis=1)

def get_compare_metric(data, predict_data, start, end):
        r_value_0 = r2_score(data[0][start:end],predict_data[0][start:end])
        r_value_1 = r2_score(data[1][start:end],predict_data[1][start:end])
        r_value_2 = r2_score(data[2][start:end],predict_data[2][start:end])
        r_value_3 = r2_score(data[0][start:end] - data[1][start:end] - data[2][start:end],
                            predict_data[0][start:end] - predict_data[1][start:end] - predict_data[2][start:end])
        print("R2:")
        print("Total infectious:", r_value_0)
        print("Daily infectious", r_value_3)
        print("Recovered: ", r_value_1)
        print("Deceased: ", r_value_2)

        mae_0 = mean_absolute_error(data[0][start:end],predict_data[0][start:end])
        mae_1 = mean_absolute_error(data[1][start:end],predict_data[1][start:end])
        mae_2 = mean_absolute_error(data[2][start:end],predict_data[2][start:end])
        mae_3 = mean_absolute_error(data[0][start:end] - data[1][start:end] - data[2][start:end],
                            predict_data[0][start:end] - predict_data[1][start:end] - predict_data[2][start:end])

        print("MAE:")
        print("Total infectious:", mae_0)
        print("Daily infectious", mae_3)
        print("Recovered: ", mae_1)
        print("Deceased: ", mae_2)

        rmse_0 = math.sqrt(mean_squared_error(data[0][start:end],predict_data[0][start:end]))
        rmse_1 = math.sqrt(mean_squared_error(data[1][start:end],predict_data[1][start:end]))
        rmse_2 = math.sqrt(mean_squared_error(data[2][start:end],predict_data[2][start:end]))
        rmse_3 = math.sqrt(mean_squared_error(data[0][start:end] - data[1][start:end] - data[2][start:end],
                            predict_data[0][start:end] - predict_data[1][start:end] - predict_data[2][start:end]))

        print("RMSE:")
        print("Total infectious:", rmse_0)
        print("Daily infectious", rmse_3)
        print("Recovered: ", rmse_1)
        print("Deceased: ", rmse_2)

        mpe_0 = np.mean(1-(predict_data[0][start:end] / data[0][start:end]))
        mpe_3 = np.mean(1-((predict_data[0][start:end]-predict_data[1][start:end]-predict_data[2][start:end]) / (data[0][start:end]-predict_data[1][start:end]-predict_data[2][start:end])))
        mpe_1 = np.mean(1-(predict_data[1][start:end] / data[1][start:end]))
        mpe_2 = np.mean(1-(predict_data[2][start:end] / data[2][start:end]))

        print("MAPE:")
        print("Total infectious:", abs(mpe_0))
        print("Daily infectious", abs(mpe_3))
        print("Recovered: ", abs(mpe_1))
        print("Deceased: ", abs(mpe_2))

        return np.array([[r_value_0, r_value_3, r_value_1, r_value_2],
                        [mae_0, mae_3, mae_1, mae_2],
                        [rmse_0, rmse_3, rmse_1, rmse_2],
                        [abs(mpe_0), abs(mpe_3), abs(mpe_1), abs(mpe_2)]])

def plotParam(list_param_byu, start, end, country="world", idx=""):
    list_param_byu = np.array(list_param_byu)
    figure, axes = plt.subplots(nrows=3, ncols=1, sharex=True)

    axes.ravel()[2].set_xlabel("Days")

    axes.ravel()[0].plot(list(range(start, end)), list_param_byu[:, 0], label="beta", color="red")
    axes.ravel()[0].set_title(r"$\beta$", fontdict={'fontsize': mpl.rcParams['axes.titlesize'],
                                             'fontweight': mpl.rcParams['axes.titleweight'],
                                             'color': 'red',
                                             'verticalalignment': 'baseline',
                                             'horizontalalignment': 'center'})

    axes.ravel()[1].plot(list(range(start, end)), list_param_byu[:, 1], label="gamma", color="blue")
    axes.ravel()[1].set_title(r"$\gamma$", fontdict={'fontsize': mpl.rcParams['axes.titlesize'],
                                             'fontweight': mpl.rcParams['axes.titleweight'],
                                             'color': 'blue',
                                             'verticalalignment': 'baseline',
                                             'horizontalalignment': 'center'})


    axes.ravel()[2].plot(list(range(start, end)), list_param_byu[:, 2], label="muy", color="gray")
    axes.ravel()[2].set_title(r"$\mu$", fontdict={'fontsize': mpl.rcParams['axes.titlesize'],
                                             'fontweight': mpl.rcParams['axes.titleweight'],
                                             'color': 'gray',
                                             'verticalalignment': 'baseline',
                                             'horizontalalignment': 'center'})

    plt.tight_layout()

    if country:
        country += "/"

    plt.savefig('images/%splot_param%s.png'%(country,idx))
    plt.close()

def plot(data, predict_data, start, end, country="world", idx=""):
    if country:
        if not os.path.exists('images/%s'%country):
            os.makedirs('images/%s'%country)

        country += "/"

    if not os.path.exists('images/%s%s'%(country,idx)):
        os.makedirs('images/%s%s'%(country,idx))

    #################################################

    fig, ax1 = plt.subplots(1,1)
    fig.suptitle('Dự báo ca nhiễm theo ngày')
    ax1.set_title(country)
    ax1.set_xlabel("Ngày")
    ax1.set_ylabel("Ca nhiễm")

    predict_plot = predict_data[1][start-1:] + predict_data[2][start-1:] + predict_data[3][start-1:]
    predict_plot = predict_plot[1:] - predict_plot[:-1]
    ax1.plot(list(range(start, start+len(predict_plot))), predict_plot, label="Dự báo")

    real_plot = data[1][start-1:end] + data[1][start-1:end] + data[2][start-1:end]
    real_plot = real_plot[1:] - real_plot[:-1]
    ax1.plot(list(range(start, end)), real_plot, label="Thực tế")

    print('Actual\tPredicted')
    print(*[str(x) + '\t' + str(y) for x,y in zip(real_plot,predict_plot[:len(real_plot)])][10:],sep='\n')
    print('Next days')
    print(predict_plot[len(real_plot):])

    length = len(predict_plot) + 5

    x = np.arange(start, start+length)
    xticks = get_list_date(length=length)

    ax1.set(xticks=x[::5], xticklabels=xticks[::5])
    loc = plticker.MultipleLocator(base=5) # this locator puts ticks at regular intervals
    ax1.xaxis.set_major_locator(loc)

    plt.legend()
    plt.savefig('images/' + country + idx + '/plot_daily_infectious.png')
    plt.close()

    write_file('images/' + country + idx + '/daily_infectious.csv', xticks[3:], predict_plot)

    #################################################

    fig3, ax3 = plt.subplots(1,1)
    fig3.suptitle('Dự báo ca hồi phục tích lũy')
    ax3.set_title(country)
    ax3.set_xlabel("Ngày")
    ax3.set_ylabel("Ca")

    predict_plot = predict_data[2][start:]
    ax3.plot(list(range(start, start+len(predict_plot))), predict_plot, label="Dự báo")

    real_plot = data[2][start:end]
    ax3.plot(list(range(start, end)), real_plot, label="Thực tế")

    length = len(predict_plot) + 5

    x = np.arange(start, start+length)
    xticks = get_list_date(length=length)

    ax3.set(xticks=x[::5], xticklabels=xticks[::5])
    loc = plticker.MultipleLocator(base=5) # this locator puts ticks at regular intervals
    ax3.xaxis.set_major_locator(loc)

    plt.legend()
    plt.savefig('images/' + country + idx + '/plot_total_recovered.png')
    plt.close()

    write_file('images/' + country + idx + '/total_recovered.csv', xticks[3:], predict_plot)

    #################################################

    fig4, ax4 = plt.subplots(1,1)
    fig4.suptitle('Dự báo ca tử vong tích lũy')
    ax4.set_title(country)
    ax4.set_xlabel("Ngày")
    ax4.set_ylabel("Ca")

    predict_plot = predict_data[3][start:]
    ax4.plot(list(range(start, start+len(predict_plot))), predict_plot, label="Dự báo")

    real_plot = data[3][start:end]
    ax4.plot(list(range(start, end)), real_plot, label="Thực tế")

    length = len(predict_plot) + 5

    x = np.arange(start, start+length)
    xticks = get_list_date(length=length)

    ax4.set(xticks=x[::5], xticklabels=xticks[::5])
    loc = plticker.MultipleLocator(base=5) # this locator puts ticks at regular intervals
    ax4.xaxis.set_major_locator(loc)

    plt.legend()
    plt.savefig('images/' + country + idx + '/plot_total_deceased.png')
    plt.close()

    write_file('images/' + country + idx + '/total_deceased.csv', xticks[3:], predict_plot)

    #################################################

    fig5, ax5 = plt.subplots(1,1)
    fig5.suptitle('Dự báo số F0 ngoài cộng đồng chưa phát hiện')
    ax5.set_title(country)
    ax5.set_xlabel("Ngày")
    ax5.set_ylabel("Ca")

    predict_plot = predict_data[0][start:]
    ax5.plot(list(range(start, start+len(predict_plot))), predict_plot, label="Dự báo")

    # real_plot = data[0][start:end]
    # ax5.plot(list(range(start, end)), real_plot, label="Thực tế")

    length = len(predict_plot) + 5

    x = np.arange(start, start+length)
    xticks = get_list_date(length=length)

    ax5.set(xticks=x[::5], xticklabels=xticks[::5])
    loc = plticker.MultipleLocator(base=5) # this locator puts ticks at regular intervals
    ax5.xaxis.set_major_locator(loc)

    plt.legend()
    plt.savefig('images/' + country + idx + '/plot_remaining_F0.png')
    plt.close()

    write_file('images/' + country + idx + '/total_remaining_F0.csv', xticks[3:], predict_plot)

def write_file(filename, days, predict_plot):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Date,Predict\n")
        for day, number in zip(days, predict_plot):
            f.write("%s,%d\n" % (day, int(number)))

def get_all_compare(data, ml_model, start, end, step=1, day_lag=10):
    print("****** Our Model ******")
    predict_data = get_predict_result(ml_model, data, start, end=end, step=step, day_lag=day_lag)
    get_compare_metric(data, predict_data, start, end)

    print("****** ARIMA ******")
    predict_data = predict_arima(data, start, end, step=step)
    get_compare_metric(data, predict_data, start, end)

    print("****** Ridge ******")
    predict_data = predict_ridge(data, start, end, step=step, day_lag=day_lag)
    get_compare_metric(data, predict_data, start, end)

    print("****** Lasso ******")
    predict_data = predict_lasso(data, start, end, step=step, day_lag=day_lag)
    get_compare_metric(data, predict_data, start, end)

    print("****** SVR ******")
    predict_data = predict_svr(data, start, end, step=step, day_lag=day_lag)
    get_compare_metric(data, predict_data, start, end)

    print("****** Decision Tree Regressor ******")
    predict_data = predict_dtr(data, start, end, step=step, day_lag=day_lag)
    get_compare_metric(data, predict_data, start, end)

    print("****** Random Forest Regressor ******")
    predict_data = predict_rfr(data, start, end, step=step, day_lag=day_lag)
    get_compare_metric(data, predict_data, start, end)

    print("****** Gradient Boosting Regressor ******")
    predict_data = predict_gbr(data, start, end, step=step, day_lag=day_lag)
    get_compare_metric(data, predict_data, start, end)

def predict_arima(data, start, end, step):
    if step > (end-start): return []
    sd = datetime.strptime("2020-01-22", "%Y-%m-%d")
    date_series = []
    for i in range(start):
        date_series.append(sd+timedelta(days=i))

    confirmed = [0 for _ in range(end-start)]
    recovered = [0 for _ in range(end-start)]
    deceased = [0 for _ in range(end-start)]
    list_div = [0 for _ in range(end-start)]

    for ste in range(start, end-step+1):
        arima_0 = ARIMA(data[0][:ste], order=(1,0,0), dates=date_series[:ste], freq="D")
        arima_1 = ARIMA(data[1][:ste], order=(1,0,0), dates=date_series[:ste], freq="D")
        arima_2 = ARIMA(data[2][:ste], order=(1,0,0), dates=date_series[:ste], freq="D")

        arima_result_0 = arima_0.fit(disp=0)
        arima_result_1 = arima_1.fit(disp=0)
        arima_result_2 = arima_2.fit(disp=0)

        a = arima_result_0.predict(start=(ste), end=(ste+step-1), dynamic=False)
        b = arima_result_1.predict(start=(ste), end=(ste+step-1), dynamic=False)
        c = arima_result_2.predict(start=(ste), end=(ste+step-1), dynamic=False)

        for i, x in enumerate(a):
            confirmed[ste-start+i] += x
            list_div[ste-start+i] += 1
        for i, x in enumerate(b):
            recovered[ste-start+i] += x
        for i, x in enumerate(c):
            deceased[ste-start+i] += x

    confirmed = np.array(confirmed) / np.array(list_div)
    recovered = np.array(recovered) / np.array(list_div)
    deceased = np.array(deceased) / np.array(list_div)

    return np.array([
                        np.append(data[0][:start], confirmed),
                        np.append(data[1][:start], recovered),
                        np.append(data[2][:start], deceased)
                    ])

def predict_ridge(data, start, end, step, day_lag):
    if step > (end-start): return []
    predict_data = []

    Ridge_model = Ridge(solver="svd")
    for f in range(data.shape[0]):
        train_data = data[f]

        res = [0 for _ in range(end-start)]
        list_div = [0 for _ in range(end-start)]
        for ste in range(start, end-step+1):
            X = []
            y = []
            for i in range(ste-day_lag):
                X.append(train_data[i:i+day_lag])
                y.append([train_data[i+day_lag]])

            Ridge_model.fit(X, y)
            tem_predict = data[f,:ste]

            for day in range(ste, ste+step):
                next_day = Ridge_model.predict([tem_predict[day-day_lag:day]])
                tem_predict = np.append(tem_predict, next_day)

            for i, x in enumerate(tem_predict[ste:]):
                res[ste-start+i] += x
                list_div[ste-start+i] += 1

        predict_data.append(np.array(res) / np.array(list_div))

    return np.append(data[:,:start], predict_data, axis=1)

def predict_lasso(data, start, end, step, day_lag):
    if step > (end-start): return []
    predict_data = []

    Lasso_model = Lasso()
    for f in range(data.shape[0]):
        train_data = data[f]

        res = [0 for _ in range(end-start)]
        list_div = [0 for _ in range(end-start)]
        for ste in range(start, end-step+1):
            X = []
            y = []
            for i in range(ste-day_lag):
                X.append(train_data[i:i+day_lag])
                y.append([train_data[i+day_lag]])

            Lasso_model.fit(X, y)
            tem_predict = data[f,:ste]

            for day in range(ste, ste+step):
                next_day = Lasso_model.predict([tem_predict[day-day_lag:day]])
                tem_predict = np.append(tem_predict, next_day)

            for i, x in enumerate(tem_predict[ste:]):
                res[ste-start+i] += x
                list_div[ste-start+i] += 1

        predict_data.append(np.array(res) / np.array(list_div))

    return np.append(data[:,:start], predict_data, axis=1)

def predict_svr(data, start, end, step, day_lag):
    if step > (end-start): return []
    predict_data = []

    SVR_model = SVR(kernel="rbf")
    for f in range(data.shape[0]):
        train_data = data[f]

        res = [0 for _ in range(end-start)]
        list_div = [0 for _ in range(end-start)]
        for ste in range(start, end-step+1):
            X = []
            y = []
            for i in range(ste-day_lag):
                X.append(train_data[i:i+day_lag])
                y.append([train_data[i+day_lag]])

            sc_X = StandardScaler()
            sc_y = StandardScaler()
            X = sc_X.fit_transform(X)
            y = sc_y.fit_transform(y)

            SVR_model.fit(X, y)
            tem_predict = data[f,:ste]

            for day in range(ste, ste+step):
                next_day = SVR_model.predict([tem_predict[day-day_lag:day]])
                next_day = sc_y.inverse_transform(next_day)
                tem_predict = np.append(tem_predict, next_day)

            for i, x in enumerate(tem_predict[ste:]):
                res[ste-start+i] += x
                list_div[ste-start+i] += 1

        predict_data.append(np.array(res) / np.array(list_div))

    return np.append(data[:,:start], predict_data, axis=1)

def predict_dtr(data, start, end, step, day_lag):
    if step > (end-start): return []
    predict_data = []

    DTR_model = DecisionTreeRegressor()
    for f in range(data.shape[0]):
        train_data = data[f]

        res = [0 for _ in range(end-start)]
        list_div = [0 for _ in range(end-start)]
        for ste in range(start, end-step+1):
            X = []
            y = []
            for i in range(ste-day_lag):
                X.append(train_data[i:i+day_lag])
                y.append([train_data[i+day_lag]])

            DTR_model.fit(X, y)
            tem_predict = data[f,:ste]

            for day in range(ste, ste+step):
                next_day = DTR_model.predict([tem_predict[day-day_lag:day]])
                tem_predict = np.append(tem_predict, next_day)

            for i, x in enumerate(tem_predict[ste:]):
                res[ste-start+i] += x
                list_div[ste-start+i] += 1

        predict_data.append(np.array(res) / np.array(list_div))

    return np.append(data[:,:start], predict_data, axis=1)

def predict_rfr(data, start, end, step, day_lag):
    if step > (end-start): return []
    predict_data = []

    RFR_model = RandomForestRegressor()
    for f in range(data.shape[0]):
        train_data = data[f]

        res = [0 for _ in range(end-start)]
        list_div = [0 for _ in range(end-start)]
        for ste in range(start, end-step+1):
            X = []
            y = []
            for i in range(ste-day_lag):
                X.append(train_data[i:i+day_lag])
                y.append([train_data[i+day_lag]])

            RFR_model.fit(X, y)
            tem_predict = data[f,:ste]

            for day in range(ste, ste+step):
                next_day = RFR_model.predict([tem_predict[day-day_lag:day]])
                tem_predict = np.append(tem_predict, next_day)

            for i, x in enumerate(tem_predict[ste:]):
                res[ste-start+i] += x
                list_div[ste-start+i] += 1

        predict_data.append(np.array(res) / np.array(list_div))

    return np.append(data[:,:start], predict_data, axis=1)

def predict_gbr(data, start, end, step, day_lag):
    if step > (end-start): return []
    predict_data = []

    GBR_model = GradientBoostingRegressor()
    for f in range(data.shape[0]):
        train_data = data[f]

        res = [0 for _ in range(end-start)]
        list_div = [0 for _ in range(end-start)]
        for ste in range(start, end-step+1):
            X = []
            y = []
            for i in range(ste-day_lag):
                X.append(train_data[i:i+day_lag])
                y.append([train_data[i+day_lag]])

            GBR_model.fit(X, y)
            tem_predict = data[f,:ste]

            for day in range(ste, ste+step):
                next_day = GBR_model.predict([tem_predict[day-day_lag:day]])
                tem_predict = np.append(tem_predict, next_day)

            for i, x in enumerate(tem_predict[ste:]):
                res[ste-start+i] += x
                list_div[ste-start+i] += 1

        predict_data.append(np.array(res) / np.array(list_div))

    return np.append(data[:,:start], predict_data, axis=1)
