def process_form_1(df):
    assert df.shape == (11, 2)
    result = {}
    today = df.loc[0][1]
    today = today.replace(hour=18)
    today = today.strftime('%b %d, %Y %H:%M')
    result['date'] = today
    pcr, rapid_test = df.loc[1, 1].split('+')
    result['new-infections'] = {'pcr': int(pcr), 'rapid-test': int(rapid_test)}
    result['total-cases'] = df.loc[2, 1]
    h, q = df.loc[3, 1].split('+')
    result['recovered'] = {'hospital': int(h), 'quarantine': int(q)}
    h, q = df.loc[4, 1].split('+')
    result['total-recovered'] = {'hospital': int(h), 'quarantine': int(q)}
    h, q = df.loc[5, 1].split('+')
    result['treatment'] = {'new-hospital': int(h), 'current': int(q)}
    a, b, c = df.loc[6, 1].split('+')
    result['critical'] = {'invasive-ventilation': int(a), 'emo': int(b)}
    result['death'] = int(c)
    result['F0'] = {'home': df.loc[7, 1], 'quarantine': df.loc[8, 1]}
    a, b = df.loc[9, 1].split('+')
    result['new-vaccine'] = {'first-dose': int(a), 'second-dose': int(b)}
    a, b = df.loc[10, 1].split('+')
    result['total-vaccine'] = {'first-dose': int(a), 'second-dose': int(b)}
    return result
def process_form_2(df):
    assert df.shape == (24, 2)
    result = {}
    today = df.loc[0][1]
    today = today.replace(hour=18)
    today = today.strftime('%b %d, %Y %H:%M')
    result['date'] = today
    districts = ['THU DUC'] + ['QUAN ' + str(i) for i in [1, 3, 4, 5, 6, 7, 8, 10, 11, 12]] +['BINH THANH', 'BINH TAN', 'PHU NHUAN', 'TAN BINH', 'TAN PHU', 'GO VAP', 'CU CHI', 'CAN GIO', 'BINH CHANH', 'HOC MON', 'NHA BE']
    for i, district in enumerate(districts):
        result[district] = df.iloc[i + 2, 1]
    return result
def process_form_3(df):
    assert df.shape == (18, 4)
    result = {}
    today = df.loc[0][1]
    today = today.replace(hour=18)
    today = today.strftime('%b %d, %Y %H:%M')
    result['date'] = today
    data = df.fillna(0).values[2:, 1:].astype(int).tolist()
    result['data'] = data
    return result