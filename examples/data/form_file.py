import numpy as np
import pandas as pd
import pickle
import time

fl = pd.read_csv('tep_data.csv')
start_d = int(time.mktime(time.strptime('2021-10-28 00:00:00', '%Y-%m-%d %H:%M:%S')))
end_d = int(time.mktime(time.strptime('2021-10-30 00:00:00', '%Y-%m-%d %H:%M:%S')))

grid = np.linspace(start_d, end_d, len(fl))


with open('myts_samples.pkl', 'wb') as f:
    for key in fl.keys():
        sl = {"grid": grid, "target": fl[key]}
        pickle.dump(sl, f)


