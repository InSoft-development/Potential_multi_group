import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
from dateutil import parser as parse_date
import argparse

import time

import os
import json
import clickhouse_connect
import sqlite3


DATA_DIR = f'Data'


def create_parser():
    parser = argparse.ArgumentParser(description="calculate time to anomaly through probability")
    parser.add_argument("-v", "--version", action="version", help="print version", version="1.0.1")
    return parser


def rolling_probability(df):
    # Первые индексы после сглаживания будут Nan, запоминаем их
    temp_rows = df['P'].iloc[:config_json['model']['rolling']*config_json['number_of_samples']]
    rolling_prob = df['P'].rolling(window=config_json['model']['rolling']*config_json['number_of_samples']).mean()
    rolling_prob.iloc[:config_json['model']['rolling']*config_json['number_of_samples']] = temp_rows
    df['P'] = rolling_prob
    return df


def calculate_anomaly_time_all_df(path_to_csv, path_to_probability, path_to_anomaly_time):
    source_data = config_json["source_input_data"]
    if source_data == "clickhouse":
        print("source from clickhouse")
        client = clickhouse_connect.get_client(host='10.23.0.177', username='default', password='asdf')
        df_csv = client.query_df(f"{config_json['paths']['database']['clickhouse']['original_csv_query']}")
        df_csv = df_csv['timestamp']
        df_csv = pd.DataFrame(df_csv, columns=['timestamp'])
        df_csv['timestamp'] = df_csv['timestamp'].astype('object')
        client.close()
    elif source_data == "sqlite":
        print("source from sqlite")
        client = sqlite3.connect(f"{config_json['paths']['database']['sqlite']['original_csv']}")
        df_csv = pd.read_sql_query(f"{config_json['paths']['database']['sqlite']['original_csv_query']}", client)
        df_csv = df_csv['timestamp']
        df_csv = pd.DataFrame(df_csv, columns=['timestamp'])
        df_csv['timestamp'] = df_csv['timestamp'].astype('object')
        client.close()
    elif source_data == "csv":
        print("source from csv")
        df_csv = pd.read_csv(path_to_csv, usecols=['timestamp'])
    else:
        print("complete field source_input_data in config (possible value: clickhouse, sqlite, csv) and rerun script")
        exit(0)
    df_probability = pd.read_csv(path_to_probability)

    df_prediction_time = pd.DataFrame(
        columns=['timestamp', 'N', 'potential', 'KrP', 'P', 'anomaly_time', 'KrT', 'anomaly_date'],
        data={'timestamp': df_csv['timestamp']})
    end_regression = []

    df_csv = pd.merge(df_csv, df_probability, how='left', on='timestamp')
    print(df_csv)
    # Сглаживание вероятности
    df_csv = rolling_probability(df_csv)
    print(df_csv)

    # Критерии по достижению вероятности и прогнозируемому времени
    delta_tau_P = 0
    delta_tau_T = 0
    freeze = False

    # Сохранение коэффициентов регрессии
    models_json = {}

    start = time.time()
    for index, row in df_csv.iterrows():
        if pd.isna(row['P']) or not(pd.notna(row['P'])):
            row['P'] = df_prediction_time.iloc[index-1]['P']
            row['N'] = df_prediction_time.iloc[index-1]['N']
            row['potential'] = df_prediction_time.iloc[index-1]['potential']

            delta_tau_P = 0
            delta_tau_T = 0
            freeze = True
        else:
            freeze = False
        df_prediction_time.iloc[index]['P'] = row['P']
        df_prediction_time.iloc[index]['N'] = row['N']
        df_prediction_time.iloc[index]['potential'] = row['potential']

        # если уже в аномалии
        if row['P'] >= config_json['model']['P_pr'] * 100:
            print(row['timestamp'], row['P'], "ANOMALY")
            df_prediction_time.iloc[index]['anomaly_time'] = "0"
            df_prediction_time.iloc[index]['anomaly_date'] = row['timestamp'][:len(row['timestamp']) - 3]

            # Критерий по достижению вероятности
            delta_tau_P += 1
            # Критерий по достижению прогнозируемого времени
            delta_tau_T = 0
            df_prediction_time.iloc[index]['KrT'] = "0"
            if (delta_tau_P > (config_json['model']['delta_tau_P']*config_json['number_of_samples'])) and (not freeze):
                df_prediction_time.iloc[index]['KrP'] = "1"
            else:
                df_prediction_time.iloc[index]['KrP'] = "0"
            # Критерий по достижению прогнозируемого времени
            '''if delta_tau_T > 144:
                df_prediction_time.iloc[index]['KrT'] = "1"
            else:
                df_prediction_time.iloc[index]['KrT'] = "0"'''
        else:
            regress_days = config_json['model']['delta'] * config_json['number_of_samples']  # окно - период

            # Критерий по достижению вероятности
            delta_tau_P = 0
            df_prediction_time.iloc[index]['KrP'] = "0"

            # Если данных для окна хватает
            if index > regress_days:
                x = np.array(range(index - regress_days, index, config_json['model']['s'])).reshape((-1, 1))  # x в окне
                y = np.array(df_prediction_time[index - regress_days:index:config_json['model']['s']]['P'])  # вероятность
                start_regression = time.time()
                model = LinearRegression().fit(x, y)
                end_regression.append(time.time() - start_regression)
                models_json[row['timestamp']] = ({"k": model.coef_[0], "b": model.intercept_})
                root = (config_json['model']['P_pr'] * 100 - df_csv.iloc[index]['P']) / model.coef_[0]  # нахождение корня. Вероятность = 95%
                print("work regression")
                # если лин коэф положительный и корень раньше, чем через 3 месяца
                if (model.coef_[0] > 0) and ((root / config_json['number_of_samples']) < 720):
                    # Корень либо внутри окна,либо правее него
                    # если внутри окна то ноль так как рост аномалии указывал на момент времени в прошлом
                    # Аномалия могла уже наступить
                    date_time = parse_date.parse(row['timestamp'])
                    df_prediction_time.iloc[index]['anomaly_time'] = \
                        0.0 if max(0, (root - float(index)) / config_json['number_of_samples']) == 0 else (
                                (root - float(index)) / config_json['number_of_samples'])
                    df_prediction_time.iloc[index]['anomaly_date'] = date_time + datetime.timedelta(
                        hours=(root) / config_json['number_of_samples'])
                    df_prediction_time.iloc[index]['anomaly_date'] = df_prediction_time.iloc[index][
                        'anomaly_date'].strftime("%Y-%m-%d %H:%M:%S")

                    # Критерий по достижению прогнозируемого времени
                    if (datetime.timedelta(hours=df_prediction_time.iloc[index]['anomaly_time']) <=
                            datetime.timedelta(hours=config_json['model']['T_pr'])):
                        delta_tau_T += 1
                        if (delta_tau_T > (config_json['model']['delta_tau_T'] * config_json['number_of_samples'])) \
                                and (not freeze):
                            df_prediction_time.iloc[index]['KrT'] = "1"
                        else:
                            df_prediction_time.iloc[index]['KrT'] = "0"
                    else:
                        delta_tau_T = 0
                        df_prediction_time.iloc[index]['KrT'] = "0"
                    print(row['timestamp'], row['P'], (root) / config_json['number_of_samples'])
                else:
                    # либо вероятно падает либо до нее > 3 месяцев
                    df_prediction_time.iloc[index]['anomaly_time'] = "NaN"
                    df_prediction_time.iloc[index]['anomaly_date'] = "NaN"
                    df_prediction_time.iloc[index]['KrT'] = "0"
                    delta_tau_T = 0
                    print(row['timestamp'], row['P'], "N/A")
            else:
                # Недостаточно данных для формирования окна
                df_prediction_time.iloc[index]['anomaly_time'] = "NaN"
                df_prediction_time.iloc[index]['anomaly_date'] = "NaN"
                df_prediction_time.iloc[index]['KrT'] = "0"
                delta_tau_T = 0
                print(row['timestamp'], row['P'], "Window...")
    end = time.time() - start
    # сохранение коэффициентов в json
    path_to_save_models = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                          f"{config_json['paths']['files']['save_models_prob']}{group}.json"
    with open(path_to_save_models, 'w', encoding='utf8') as f:
        json.dump(models_json, f, ensure_ascii=False, indent=4)

    df_prediction_time.to_csv(path_to_anomaly_time, index=False)
    print("len of df: ", len(df_csv))
    print("time of code : ", end, " seconds")
    print("average time of regression: ", sum(end_regression) / len(end_regression), " seconds")


if __name__ == '__main__':
    # Заполнение коэффициентов json из всего dataframe
    parser = create_parser()
    namespace = parser.parse_args()
    with open("config_SOCHI.json", 'r', encoding='utf8') as j:
        config_json = json.load(j)
    print("config SOCHI")
    path_to_csv = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['original_csv']}"
    with open(f"{DATA_DIR}{os.sep}{config_json['paths']['files']['json_sensors']}", 'r', encoding='utf8') as f:
        json_dict = json.load(f)

    index_group = [list(x.keys())[0] for x in json_dict["groups"]]
    if index_group[0] == '0':
        index_group.remove('0')
    for group in index_group:
        path_to_probability = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                              f"{config_json['paths']['files']['probability_csv']}{group}.csv"
        path_to_potentials = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                             f"{config_json['paths']['files']['potentials_csv']}{group}.csv"
        path_to_anomaly_time = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                               f"{config_json['paths']['files']['anomaly_time_prob']}{group}.csv"
        calculate_anomaly_time_all_df(path_to_csv, path_to_probability, path_to_anomaly_time)
