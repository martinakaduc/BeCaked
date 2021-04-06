from __future__ import division
import numpy as np
import pandas as pd
import os

class DataLoader():
    def __init__(self, folder="COVID-19/csse_covid_19_data/csse_covid_19_time_series/"):
        self.confirmed_obj = pd.read_csv(os.path.join(folder, "time_series_covid19_confirmed_global.csv"))
        self.deaths_obj = pd.read_csv(os.path.join(folder, "time_series_covid19_deaths_global.csv"))
        self.recovered_obj = pd.read_csv(os.path.join(folder, "time_series_covid19_recovered_global.csv"))

        self.confirmed = [sum(self.confirmed_obj[key]) for key in self.confirmed_obj.keys()[4:]]
        self.deaths = [sum(self.deaths_obj[key]) for key in self.deaths_obj.keys()[4:]]
        self.recovered = [sum(self.recovered_obj[key]) for key in self.recovered_obj.keys()[4:]]
        self.total_day = len(self.confirmed_obj.keys()) - 4

        countries_temp = self.confirmed_obj.values[:,:4]
        self.countries = [list(set([x[1] for x in countries_temp])), []]

        for x in self.countries[0]:
            list_loc = np.where(countries_temp == x)[0]
            lat_long = countries_temp[list_loc][:,-2:]
            mean_loc = np.nanmean(lat_long, axis=0)
            self.countries[1].append(mean_loc.tolist())

    def get_data_world_series(self):
        return np.array([np.array(self.confirmed), np.array(self.recovered), np.array(self.deaths)], dtype=np.float64)

    def get_data_countries_series(self):
        confirmed = {x:np.zeros((self.total_day,), dtype=int) for x in self.countries[0]}
        recovered = {x:np.zeros((self.total_day,), dtype=int) for x in self.countries[0]}
        deaths = {x:np.zeros((self.total_day,), dtype=int) for x in self.countries[0]}

        for x in self.confirmed_obj.values:
            confirmed[x[1]] = confirmed[x[1]] + x[4:]

        for x in self.recovered_obj.values:
            recovered[x[1]] = recovered[x[1]] + x[4:]

        for x in self.deaths_obj.values:
            deaths[x[1]] = deaths[x[1]] + x[4:]

        return [confirmed, recovered, deaths]

    def get_data_countries_current(self):
        confirmed = {x:0 for x in self.countries[0]}
        recovered = {x:0 for x in self.countries[0]}
        deaths = {x:0 for x in self.countries[0]}

        for x in self.confirmed_obj.values:
            confirmed[x[1]] += x[-1]

        for x in self.recovered_obj.values:
            recovered[x[1]] += x[-1]

        for x in self.deaths_obj.values:
            deaths[x[1]] += x[-1]

        return [confirmed, recovered, deaths]

    def get_data_countries_increase(self):
        confirmed = {x:0 for x in self.countries[0]}
        recovered = {x:0 for x in self.countries[0]}
        deaths = {x:0 for x in self.countries[0]}

        for x in self.confirmed_obj.values:
            confirmed[x[1]] += x[-1] - x[-2]

        for x in self.recovered_obj.values:
            recovered[x[1]] += x[-1] - x[-2]

        for x in self.deaths_obj.values:
            deaths[x[1]] += x[-1] - x[-2]

        return [confirmed, recovered, deaths]

    def get_countries(self):
        return self.countries

    def get_current_day(self):
        day_time= self.confirmed_obj.keys()[-1].split('/')
        return '%s-%s-%s' % ('20'+day_time[2], '%.2d'%int(day_time[0]), day_time[1])
