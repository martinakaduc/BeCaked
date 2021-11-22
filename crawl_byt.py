import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
import base64
import glob
import time

def crawl_byt_by_id(id):
    id = str(id).zfill(2)
    url = 'https://forms.datagalaxy.dev/covid/export-excel/'
    payload = {"id":id,"date":datetime.now().replace(microsecond=0).strftime("%d/%m/%Y %H:%M:%S")}
    headers = {'Content-type': 'application/json;charset=UTF-8'}
    iter = 1
    while 1:
        try:
            print('ID={}'.format(id))
            response = requests.post(url, json = payload, headers = headers, timeout=10, verify=False)
            break
        except Exception as e:
            print(e)
            iter += 1
            print("{}: {} time".format(id,iter))
    enc = json.loads(response.text)["result"]["data"]
    dec = base64.b64decode(enc)
    return dec

def crawl_byt(to_dir):
    ids = ['48', '75', '87', '67', '66', '11', '89', '77', '52', '74', '70', '60', '95', '24', '6', '27', '83', '96', '4', '92', '64', '2', '1', '35', '42', '79', '17', '33', '30', '31', '93', '56', '91', '62', '10', '67', '12', '20', '80', '36', '40', '37', '58', '25', '54', '44', '49', '51', '22', '45', '94', '14', '72', '34', '19', '46', '38', '82', '84', '8', '86', '26', '15']
    for id in ids:
        with open(os.path.join(to_dir,id+'.xlsx'),'wb') as f:
            f.write(crawl_byt_by_id(id))
    process_byt_all(to_dir)

def process_byt_one(filepath):
    note = pd.read_excel(filepath).iloc[0,0].split(' ')[-2] #last update: "(Cập nhật ngày 12/11/2021 )"
    df = pd.read_excel(filepath,usecols=[2,4],skiprows=8)
    k = df.iloc[:,0].tolist()
    v = df.iloc[:,1].tolist()
    v = [int(x[-1]) for x in v]
    return {x:(y,note) for x,y in zip(k,v)}

def process_byt_all(dirpath):
    result = {}
    for filepath in glob.glob(os.path.join(dirpath,'*.xlsx')):
        new = process_byt_one(filepath)
        result = {**result,**new}
    with open(os.path.join(dirpath,'levels.json'),'w') as f:
        json.dump(result,f)


if __name__ == '__main__':
    backup_data_dir = os.environ.get("BACKUP_DATA_PATH", "./backup/")
    crawl_byt(to_dir=os.path.join(backup_data_dir,'byt'))
