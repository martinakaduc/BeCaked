from __future__ import division
import numpy as np
import pandas as pd
import os

class DataLoader():
    def __init__(self, folder="COVID-19/HCM/", ward_name="HCM"):
        self.infectious_obj = pd.read_excel(os.path.join(folder, "SEIRD_data_12_7_2021.xlsx"), sheet_name="Infectious")
        self.deaths_obj = pd.read_excel(os.path.join(folder, "SEIRD_data_12_7_2021.xlsx"), sheet_name="Deaths")
        self.recovered_obj = pd.read_excel(os.path.join(folder, "SEIRD_data_12_7_2021.xlsx"), sheet_name="Recovered")
        self.exposed_obj = pd.read_excel(os.path.join(folder, "SEIRD_data_12_7_2021.xlsx"), sheet_name="Exposed")
        self.N_obj = pd.read_excel(os.path.join(folder, "SEIRD_data_12_7_2021.xlsx"), sheet_name="Population")

        self.exposed = self.exposed_obj[ward_name].to_numpy()
        self.infectious = self.infectious_obj[ward_name].to_numpy()
        self.deaths = self.deaths_obj[ward_name].to_numpy()
        self.recovered = self.recovered_obj[ward_name].to_numpy()
        self.beta = np.zeros_like(self.infectious,dtype=self.infectious.dtype)
        self.N = self.N_obj[ward_name].to_numpy()

        self.modify()

        self.total_day = len(self.infectious)
        print(self.infectious_obj.Date.to_list()[-1])

    def modify(self):
        Ie_acc,Is_acc,R_acc,D_acc = self.exposed.copy(), self.infectious.copy(), self.recovered.copy(), self.deaths.copy()
        for i in range(1,len(self.exposed)):
            Ie_acc[i] += Ie_acc[i-1]    #Ie = a+b
            Is_acc[i] += Is_acc[i-1]    #Is = c+d
            R_acc[i] += R_acc[i-1]
            D_acc[i] += D_acc[i-1]

        I_acc = Is_acc + Ie_acc
        # E_acc = Ie_acc[5:]
        # I_acc = I_acc[:-5]
        E = Ie_acc[5:] - Ie_acc[:-5]
        E = np.append(E, np.zeros((5,)))
        R = R_acc
        D = D_acc
        I = I_acc - R - D
        N = self.N[0] * np.ones_like(I)
        # S = self.N[0] - I - E - R - D

        self.exposed,self.infectious,self.recovered,self.deaths,self.N = E,I,R,D,N
        print(E,I,R,D,sep='\n')

    def get_data_world_series(self):
        return np.array([self.exposed, self.infectious, self.recovered, self.deaths, self.beta, self.N], dtype=np.float64)

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
        day_time= self.infectious_obj.iloc[len(self.infectious_obj) - 1]
        return '%s-%s-%s' % ('20'+day_time[2], '%.2d'%int(day_time[0]), day_time[1])
