import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
from dateutil import parser as parse_date
import argparse
import sys

import time
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os


DATA_DIR = f'Data'


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_csv", nargs=1, help="name of the original CSV file")
    parser.add_argument("path_to_probability", nargs=1, help="name of CSV file with saved probabilities")
    parser.add_argument("path_to_potentials", nargs=1, help="name of CSV file with saved potentials")
    parser.add_argument("path_to_anomaly_time", nargs=1, help="name of saving CSV file with anomaly date")
    return parser


def rolling_probability(df):
    # Первые индексы после сглаживания будут Nan, запоминаем их
    temp_rows = df['P'].iloc[:config_json['model']['rolling']*config_json['number_of_samples']]
    rolling_prob = df['P'].rolling(window=config_json['model']['rolling']*config_json['number_of_samples']).mean()
    rolling_prob.iloc[:config_json['model']['rolling']*config_json['number_of_samples']] = temp_rows
    df['P'] = rolling_prob
    return df


def regress_lines(path_to_anomaly_time):
    df_anomaly = pd.read_csv(path_to_anomaly_time, index_col=[0])
    #df_anomaly = df_anomaly.iloc[:30000]
    print(df_anomaly.head())

    anomaly_P = config_json['model']['P_pr'] * 100
    plt.rcParams["figure.figsize"] = (200, 10)
    plt.title("Anomaly time")
    plt.plot(df_anomaly['t'], df_anomaly['P'], "g-")
    plt.plot(df_anomaly['t'], df_anomaly['anomaly_time'], "b-")
    plt.axhline(y=anomaly_P, color='brown', linestyle='-')
    #plt.plot(df_anomaly['t'], df_anomaly['N'], "b-")
    df_KrT = df_anomaly.loc[df_anomaly['KrT'] == True]
    print(df_KrT)
    df_anomaly['t'] = pd.to_datetime(df_anomaly['t'], format="%Y-%m-%d %H:%M:%S")
    df_KrT['t'] = pd.to_datetime(df_KrT['t'], format="%Y-%m-%d %H:%M:%S")
    df_KrT['anomaly_date'] = pd.to_datetime(df_KrT['anomaly_date'], format="%Y-%m-%d %H:%M:%S")
    print(df_KrT)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=24))
    plt.legend(["probability", "anomaly_time", "P=" + str(anomaly_P) + "%", "regress lines"])
    #plt.savefig('Korshikova\\Отстройка от всего\\scale_potential_' + str(i) + '.png')
    plt.show()


def calculate_anomaly_time_all_df(path_to_csv, path_to_probability, path_to_potentials, path_to_anomaly_time):
    df_csv = pd.read_csv(path_to_csv, usecols=['timestamp'])
    df_csv.rename(columns={'timestamp': 't'}, inplace=True)
    #df_potentials = pd.read_csv(path_to_potentials)
    df_probability = pd.read_csv(path_to_probability, index_col=0)

    #col_time = ['timestamp']
    #col_index = df_potentials.columns[df_potentials.columns.to_list().index('index0'):].to_list()
    #count_index = len(col_index)
    #col = col_time + col_index

    #col = ['timestamp', 'index0', 'index1', 'index2', 'index3', 'index4']
    # df_probability = pd.merge(df_probability, df_potentials[col], how='inner',
    #                           left_on='t', right_on='timestamp')

    #df_probability.drop(columns=['timestamp'], inplace=True)

    # df_prediction_time = pd.DataFrame(
    #     columns=['t', 'N', 'potential', 'KrP', 'P', 'anomaly_time', 'KrT', 'anomaly_date', 'index0', 'index1', 'index2',
    #              'index3', 'index4'],
    #     data={'t': df_csv['t']})
    df_prediction_time = pd.DataFrame(
        columns=['t', 'N', 'potential', 'KrP', 'P', 'anomaly_time', 'KrT', 'anomaly_date'],
        data={'t': df_csv['t']})
    end_regression = []
    df_csv = pd.merge(df_csv, df_probability, how='left')
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
            # for i in range(0, count_index):
            #     row["index" + str(i)] = df_prediction_time.iloc[index-1]["index" + str(i)]
            delta_tau_P = 0
            delta_tau_T = 0
            freeze = True
        else:
            freeze = False
        df_prediction_time.iloc[index]['P'] = row['P']
        df_prediction_time.iloc[index]['N'] = row['N']
        df_prediction_time.iloc[index]['potential'] = row['potential']
        # for i in range(0, count_index):
        #     df_prediction_time.iloc[index]["index" + str(i)] = row["index" + str(i)]
        # если уже в аномалии
        if row['P'] >= config_json['model']['P_pr'] * 100:
            print(row['t'], row['P'], "ANOMALY")
            df_prediction_time.iloc[index]['anomaly_time'] = "0"
            df_prediction_time.iloc[index]['anomaly_date'] = row['t'][:len(row['t']) - 3]

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
                models_json[row['t']] = ({"k": model.coef_[0], "b": model.intercept_})
                root = (config_json['model']['P_pr'] * 100 - model.intercept_) / model.coef_[0]
                print("work regression")
                # если лин коэф положительный и корень раньше, чем через 3 месяца
                if (model.coef_[0] > 0) and (((root - float(index)) / config_json['number_of_samples']) < 720):
                    # Корень либо внутри окна,либо правее него
                    # если внутри окна то ноль так как рост аномалии указывал на момент времени в прошлом
                    # Аномалия могла уже наступить
                    date_time = parse_date.parse(row['t'])
                    df_prediction_time.iloc[index]['anomaly_time'] = \
                        0.0 if max(0, (root - float(index)) / config_json['number_of_samples']) == 0 else (
                                (root - float(index)) / config_json['number_of_samples'])
                    df_prediction_time.iloc[index]['anomaly_date'] = date_time + datetime.timedelta(
                        hours=(root - float(index)) / config_json['number_of_samples'])
                    df_prediction_time.iloc[index]['anomaly_date'] = df_prediction_time.iloc[index][
                        'anomaly_date'].strftime("%Y-%m-%d %H:%M")

                    # Критерий по достижению прогнозируемого времени
                    if (datetime.timedelta(hours=df_prediction_time.iloc[index]['anomaly_time']) <=
                            datetime.timedelta(hours=config_json['model']['T_pr'])):
                        delta_tau_T += 1
                        if delta_tau_T > (config_json['model']['delta_tau_T'] * config_json['number_of_samples'])\
                                and (not freeze):
                            df_prediction_time.iloc[index]['KrT'] = "1"
                        else:
                            df_prediction_time.iloc[index]['KrT'] = "0"
                    else:
                        delta_tau_T = 0
                        df_prediction_time.iloc[index]['KrT'] = "0"
                    print(row['t'], row['P'], (root - float(index)) / config_json['number_of_samples'])
                else:
                    # либо вероятно падает либо до нее > 3 месяцев
                    df_prediction_time.iloc[index]['anomaly_time'] = "NaN"
                    df_prediction_time.iloc[index]['anomaly_date'] = "NaN"
                    df_prediction_time.iloc[index]['KrT'] = "0"
                    delta_tau_T = 0
                    print(row['t'], row['P'], "N/A")
            else:
                # Недостаточно данных для формирования окна
                df_prediction_time.iloc[index]['anomaly_time'] = "NaN"
                df_prediction_time.iloc[index]['anomaly_date'] = "NaN"
                df_prediction_time.iloc[index]['KrT'] = "0"
                delta_tau_T = 0
                print(row['t'], row['P'], "Window...")
    end = time.time() - start
    # сохранение коэффициентов в json
    path_to_save_models = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                          f"{config_json['paths']['files']['save_models_intercept']}{group}.json"
    with open(path_to_save_models, 'w', encoding='utf8') as f:
        json.dump(models_json, f, ensure_ascii=False, indent=4)

    df_prediction_time.to_csv(path_to_anomaly_time)
    print("len of df: ", len(df_csv))
    print("time of code : ", end, " seconds")
    print("average time of regression: ", sum(end_regression) / len(end_regression), " seconds")


def check_regress(path_to_anomaly_time):
    df_anomaly = pd.read_csv(path_to_anomaly_time, index_col=[0])
    #df_anomaly['t'] = pd.to_datetime(df_anomaly['t'], format="%Y-%m-%d %H:%M:%S")
    #df_anomaly['anomaly_date'] = pd.to_datetime(df_anomaly['anomaly_date'], format="%Y-%m-%d %H:%M:%S")
    df_anomaly = df_anomaly.iloc[:30000]
    print(df_anomaly.head())
    current_t = df_anomaly.iloc[1084]['t']
    print(current_t)

    index = 1084
    regress_days = config_json['model']['delta'] * config_json['number_of_samples']  # 3 дня
    print(index - regress_days)

    x = np.array(range(index - regress_days, index + 1), config_json['model']['s']).reshape((-1, 1))  # x в окне
    y = np.array(df_anomaly.iloc[index - regress_days:index + 1: config_json['model']['s']]['P'])  # вероятность

    model = LinearRegression().fit(x, y)
    print("work regression")
    print(model.coef_[0], model.intercept_)

    anomaly_P = config_json['model']['P_pr'] * 100
    plt.rcParams["figure.figsize"] = (200, 10)
    plt.title("Anomaly time")
    plt.plot(df_anomaly['t'], df_anomaly['P'], "g-")
    plt.plot(df_anomaly['t'], df_anomaly['anomaly_time'], "b-")
    plt.axhline(y=anomaly_P, color='brown', linestyle='-')
    # plt.plot(df_anomaly['t'], df_anomaly['N'], "b-")
    #x_regress = np.array(range(index - regress_days, index + 1)).reshape((-1, 1))
    #x_regress = np.array(range(index - regress_days, index + regress_days + 1)).reshape((-1, 1))
    x_regress = np.array(range(index - regress_days, index + 250, config_json['model']['s'])).reshape((-1, 1))
    y_regress = model.predict(x_regress)
    y_list = []
    for i in y_regress:
        if i <= anomaly_P:
            y_list.append(i)
        if i > anomaly_P:
            y_list.append(i)
            break
    x_regress = x_regress[:len(y_list)]
    plt.plot(x_regress, y_list, color="red", linewidth=0.5)

    root = (anomaly_P - model.intercept_) / model.coef_[0]
    #root = (95 - df_anomaly.iloc[index]['P']) / model.coef_[0]
    print(root)
    print(str(datetime.datetime.strptime(current_t, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(minutes=5 * root)))
    date = datetime.datetime.strptime(current_t, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(minutes=5 * root)
    date = datetime.datetime.strftime(date, "%Y-%m-%d %H:%M:%S")
    print(date)

    similar_date = [df_anomaly.loc[df_anomaly['t'] <= date].tail(1), df_anomaly.loc[df_anomaly['t'] >= date].head(1)]
    if similar_date[1].empty:
        similar_date[1] = df_anomaly.tail(1)
    similar_date[0] = datetime.datetime.strptime(similar_date[0].iloc[0]['t'], "%Y-%m-%d %H:%M:%S")
    similar_date[1] = datetime.datetime.strptime(similar_date[1].iloc[0]['t'], "%Y-%m-%d %H:%M:%S")
    most_similar_date = min(similar_date, key=lambda x: abs(x -
                                                            datetime.datetime.strptime(current_t, "%Y-%m-%d %H:%M:%S")))

    x_predict = [str(current_t), str(most_similar_date)]  # str(most_similar_date)]
    print(x_predict)
    y_predict = [df_anomaly.iloc[1084]['P'], anomaly_P]
    print(y_predict)
    #plt.plot(x_predict, y_predict, color="red", linewidth=0.5)
    plt.axvline(x=df_anomaly.iloc[index - regress_days]['t'], color='black', linestyle='-')
    plt.axvline(x=str(current_t), color='black', linestyle='-')
    #plt.axvline(x=str(most_similar_date), color='black', linestyle='-')
    plt.axvline(x=str(df_anomaly.iloc[x_regress[len(x_regress)-1]]['t'].iloc[0]), color='black', linestyle='-')
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.legend(["probability", "anomaly_time", "P=95%", "regress lines"])
    # plt.savefig('Korshikova\\Отстройка от всего\\scale_potential_' + str(i) + '.png')
    plt.show()


if __name__ == '__main__':
    # Заполнение коэффициентов json из всего dataframe
    parser = create_parser()
    with open("config_SOCHI.json", 'r', encoding='utf8') as j:
        config_json = json.load(j)
    if len(sys.argv) == 1:
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
                                   f"{config_json['paths']['files']['anomaly_time_intercept']}{group}.csv"
            calculate_anomaly_time_all_df(path_to_csv, path_to_probability, path_to_potentials, path_to_anomaly_time)
    else:
        print("command's line arguments")
        namespace = parser.parse_args()
        path_to_csv = namespace.path_to_csv[0]
        path_to_probability = namespace.path_to_probability[0]
        path_to_potentials = namespace.path_to_potentials[0]
        path_to_anomaly_time = namespace.path_to_anomaly_time[0]
        calculate_anomaly_time_all_df(path_to_csv, path_to_probability, path_to_potentials, path_to_anomaly_time)
        regress_lines(path_to_anomaly_time)

    #check_regress(path_to_anomaly_time)
