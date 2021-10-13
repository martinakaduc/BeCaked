from pymongo import MongoClient
import glob
import os
import pandas as pd
import shutil
from datetime import datetime as dt
import sys

def remove_v2_path(folder_name = 'data'):
    all_csv_paths = glob.glob(f'{folder_name}/*/v2/*/*/*.csv')
    for path in all_csv_paths:
        new_path = path.replace('/v2', '')
        new_dir = new_path.rsplit('/', 1)[0]
        if not os.path.exists(new_dir):
            os.system(f'mkdir -p "{new_dir}"')
        os.replace(path, new_path)

def prepare_all_data(folder_name = 'data'):
    all_csv_paths = glob.glob(f'{folder_name}/*/*/*/*.csv')
    districts = ['BINH CHANH', 'BINH TAN', 'BINH THANH', 'CAN GIO', 'CU CHI', 'GO VAP', 'HCM', 'HOC MON', 'NHA BE', 'PHU NHUAN'] + [f'QUAN {i}' for i in [1, 3, 4, 5, 6, 7, 8, 10, 11, 12]] + ['TAN BINH', 'TAN PHU', 'THU DUC']
    data = {}
    for district in districts:
        paths = list(filter(lambda x: district in x, all_csv_paths))
        data[district] = {}
        for path in paths:
            coms = path.split(os.path.sep)
            date = coms[1]
            case = coms[-2]
            if os.stat(path).st_size == 0: continue
            df = pd.read_csv(path)
            name = coms[-1]
            if 'infectious' in name:
                compartment = 'I'
            elif 'recovered' in name:
                compartment = 'R'
            elif 'deceased' in name:
                compartment = 'D'
            elif 'critical' in name or 'ecmo' in name:
                compartment = 'C'
            else: continue
            if date not in data[district]:
                data[district][date] = {}
                data[district][date][compartment] = {}
                data[district][date][compartment][case] = df['Predict'].values.astype(int).tolist()
                if 'Real' in df.columns:
                    data[district][date][compartment]['real'] = df['Real'].dropna(axis = 0).values.astype(int).tolist()
                data[district][date]['dates'] = df['Date'].values.tolist()
            else:
                if compartment not in data[district][date]:
                    data[district][date][compartment] = {}
                if 'Real' in df.columns and 'real' not in data[district][date][compartment]:
                    data[district][date][compartment]['real'] = df['Real'].dropna(axis = 0).values.astype(int).tolist()
                data[district][date][compartment][case] = df['Predict'].values.astype(int).tolist()
    return data
def convert_into_mongo_format(data):
    mongo_data = {}
    for name, value in data.items():
        mongo_data[name] = []
        for date, _data in value.items():
            mongo_data[name].append({'_id': date, 'data': _data})
    return mongo_data

def refactor_date(folder):
    all_csv_paths = glob.glob(f'{folder}/*/*/*/*.csv')
    districts = ['BINH CHANH', 'BINH TAN', 'BINH THANH', 'CAN GIO', 'CU CHI', 'GO VAP', 'HCM', 'HOC MON', 'NHA BE', 'PHU NHUAN'] + [f'QUAN {i}' for i in [1, 3, 4, 5, 6, 7, 8, 10, 11, 12]] + ['TAN BINH', 'TAN PHU', 'THU DUC']
    for district in districts:
        paths = list(filter(lambda x: district in x, all_csv_paths))
        dates = set()
        for path in paths:
            dates.add(path.split(os.path.sep)[1])
        for date in dates:
            for case in ['NormalCase', 'BestCase', 'WorstCase']:
                path = f'{folder}/{date}/{district}/{case}/'
                sd_i, sd_r, sd_d = None, None, None
                try:
                    df_I = pd.read_csv(path + 'daily_infectious.csv')
                    sd_i = dt.strptime(df_I['Date'].values[0] + '/21', '%d/%m/%y')
                except: pass
                try:
                    df_R = pd.read_csv(path + 'total_recovered.csv')
                    sd_r = dt.strptime(df_R['Date'].values[0]+ '/21', '%d/%m/%y')
                except: pass
                try:
                    df_D = pd.read_csv(path + 'total_deceased.csv')
                    sd_d = dt.strptime(df_D['Date'].values[0]+ '/21', '%d/%m/%y')
                except: pass
                if sd_i is not None and sd_d is not None and sd_r is not None:
                    if sd_i == sd_r == sd_d: continue
                    m_date = max(sd_i, max(sd_d, sd_r))
                    if sd_i < m_date:
                        df_I = df_I.loc[(m_date - sd_i).days:]
                        df_I.to_csv(path + 'daily_infectious.csv', index = False)
                    if sd_d < m_date:
                        df_D = df_D.loc[(m_date - sd_d).days:]
                        df_D.to_csv(path + 'total_deceased.csv', index = False)
                    if sd_r < m_date:
                        df_R = df_R.loc[(m_date - sd_r).days:]
                        df_R.to_csv(path + 'total_recovered.csv', index = False)
                elif sd_d is not None and sd_r is not None:
                    if sd_r == sd_d: continue
                    m_date = max(sd_d, sd_r)
                    if sd_d < m_date:
                        df_D = df_D.loc[(m_date - sd_d).days:]
                        df_D.to_csv(path + 'total_deceased.csv', index = False)
                    if sd_r < m_date:
                        df_R = df_R.loc[(m_date - sd_r).days:]
                        df_R.to_csv(path + 'total_recovered.csv', index = False)
                elif sd_i is not None and sd_r is not None:
                    if sd_r == sd_i: continue
                    m_date = max(sd_i, sd_r)
                    if sd_i < m_date:
                        df_I = df_I.loc[(m_date - sd_i).days:]
                        df_I.to_csv(path + 'daily_infectious.csv', index = False)
                    if sd_r < m_date:
                        df_R = df_R.loc[(m_date - sd_r).days:]
                        df_R.to_csv(path + 'total_recovered.csv', index = False)
                elif sd_d is not None and sd_i is not None:
                    if sd_d == sd_i: continue
                    m_date = max(sd_i, sd_r)
                    if sd_i < m_date:
                        df_I = df_I.loc[(m_date - sd_i).days:]
                        df_I.to_csv(path + 'daily_infectious.csv', index = False)
                    if sd_d < m_date:
                        df_D = df_D.loc[(m_date - sd_d).days:]
                        df_D.to_csv(path + 'total_deceased.csv', index = False)


                
def query_data(db, district, date):
    return db[district].find_one({"_id": date})
def get_latest_data(db, district):
    latest_date = db.auxiliary.find_one({"type": "latest_date"}, {"latest_date": 1})['latest_date']
    return query_data(db, district, latest_date)

def insert_new_data(db, folder_name):
    data = prepare_all_data(folder_name)
    mongo_data = convert_into_mongo_format(data)
    for name, value in mongo_data.items():
        if len(value) == 0:
            print(f'MISSED {name}')
            continue
        db[name].insert_many(value)
def update_lastest_date(db, date):
    db['auxiliary'].replace_one({"type": "latest_date"}, {"type": "latest_date", "latest_date": date})

def update_cummulative_info(db):
    df = pd.read_excel('SEIRD_data_12_7_2021.xlsx', sheet_name = None)
    df_I = df['Infectious']
    districts = list(df_I.columns)[1:]
    dates = df_I['Date'].apply(lambda x: dt.strftime(x, '%m.%d')).values.tolist()
    df_I = df_I.drop('Date', axis = 1)
    df_R = df['Recovered'].drop('Date', axis = 1)
    df_D = df['Deaths'].drop('Date', axis = 1)
    df_V = df['Vaccinated'].drop('Date', axis = 1)
    df_C = df['ECMO'].drop('Date', axis = 1)

    df_R = df_R.cumsum()
    df_D = df_D.cumsum()
    data = []
    date = dates[-1]
    i = len(dates) - 1
    row = {"_id": date}
    row['I'] = {}
    row['R'] = {}
    row['D'] = {}
    row['V'] = {}
    row['C'] = {}
    for district in districts:
        row['I'][district] = int(df_I.iloc[i][district])
        row['R'][district] = int(df_R.iloc[i][district])
        row['D'][district] = int(df_D.iloc[i][district])
        row['V'][district] = int(df_V.iloc[i][district])
    row['C']['HCM'] = int(df_C.loc[i, 'HCM'])
    data.append(row)
    db['cum_data'].insert_many(data)




if __name__ == '__main__':
    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
        remove_v2_path(folder_name)

    uri = "mongodb+srv://thesis.cojlj.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"
    client = MongoClient(uri,
                     tls=True,
                     tlsCertificateKeyFile='certificates.pem')
    # client = MongoClient("mongodb+srv://thesisbecaked:thesisbecaked@thesis.cojlj.mongodb.net")
    # client = MongoClient('mongodb://localhost:27017/')
    db = client['daily-data']
    if len(sys.argv) > 1:
        refactor_date(folder_name)
        update_lastest_date(db, folder_name)
        insert_new_data(db, folder_name)
    update_cummulative_info(db)
    # print(get_latest_data(db, "QUAN 3"))
    client.close()
    