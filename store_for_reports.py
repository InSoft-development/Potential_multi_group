import os
import shutil
import json
import pandas as pd

import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="add output files for input to streamlit web-app")
    parser.add_argument("-v", "--version", action="version", help="print version", version="1.0.0")
    return parser


parser = create_parser()
namespace = parser.parse_args()

DATA_DIR = f'Data'
REPORTS_DIR = f'Reports'
JSON_INTERVAL = f'{DATA_DIR}{os.sep}json_interval{os.sep}'

CSV_PATH = f'{DATA_DIR}{os.sep}'

CSV_REPORTS_LOSS = f'{REPORTS_DIR}{os.sep}csv_loss{os.sep}'
CSV_REPORTS_PREDICT = f'{REPORTS_DIR}{os.sep}csv_predict{os.sep}'

JSON_REPORT_INTERVAL = f'{REPORTS_DIR}{os.sep}json_interval{os.sep}'

with open("config_SOCHI.json", 'r', encoding='utf8') as j:
    config_json = json.load(j)

with open(f"{DATA_DIR}{os.sep}{config_json['paths']['files']['json_sensors']}", 'r', encoding='utf8') as f:
    json_dict = json.load(f)

try:
    os.mkdir(f"{REPORTS_DIR}")
except Exception as e:
    print(e)
    print('Directory exist')
try:
    os.mkdir(f"{JSON_REPORT_INTERVAL}")
except Exception as e:
    print(e)
    print('Directory exist')

try:
    os.mkdir(f"{CSV_REPORTS_LOSS}")
except Exception as e:
    print(e)
    print('Directory exist')

try:
    os.mkdir(f"{CSV_REPORTS_PREDICT}")
except Exception as e:
    print(e)
    print('Directory exist')

index_group = [list(x.keys())[0] for x in json_dict["groups"]]
if index_group[0] == '0':
    index_group.remove('0')
for group in index_group:
    path_to_json_old = f"{JSON_INTERVAL}{config_json['paths']['files']['intervals_json']}{group}.json"
    path_to_json_new = f"{JSON_REPORT_INTERVAL}{config_json['paths']['files']['intervals_json']}{group}.json"
    shutil.copy(f'{path_to_json_old}', f'{path_to_json_new}')

    csv_path_loss = f"{CSV_PATH}{group}{os.sep}{config_json['paths']['files']['loss_csv']}{group}.csv"
    csv_path_loss_new = f"{CSV_REPORTS_LOSS}{config_json['paths']['files']['loss_csv']}{group}.csv"

    shutil.copy(f'{csv_path_loss}', f'{csv_path_loss_new}')

    csv_path_predict_old = f"{CSV_PATH}{group}{os.sep}" \
                           f"{config_json['paths']['files']['anomaly_time_intercept']}{group}.csv"
    csv_path_predict_new = f"{CSV_REPORTS_PREDICT}predict_{group}.csv"

    csv_anomaly_time = pd.read_csv(csv_path_predict_old)
    csv_anomaly_time.rename(columns={'P': 'target_value'}, inplace=True)
    csv_anomaly_time.to_csv(csv_path_predict_new, index=False)

    #shutil.copy(f'{csv_path_predict_old}', f'{csv_path_predict_new}')
