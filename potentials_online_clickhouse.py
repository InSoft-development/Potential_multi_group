import json
import os
import argparse

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import union
import normalization

import time
import datetime
import clickhouse_connect

DATA_DIR = f'Data'


def create_parser():
    parser = argparse.ArgumentParser(description="online mode of potentials method")
    parser.add_argument("-v", "--version", action="version", help="print version", version="1.0.1")
    return parser


# Загрузка KKS
def kks_online_mode(approxlist, group):
    nums = []
    index_group = [list(x.keys())[0] for x in json_dict["groups"]]
    try:
        index_group = index_group.index(str(group))
    except ValueError:
        print("Группы " + str(group) + " не существует")

    for unions in json_dict["groups"][index_group].values():
        if unions["unions"] != "null":
            for union_val in unions["unions"]:
                for element in union_val.values():
                    # Добавляем тэги union датчиков
                    nums.append(element["name"] + "_min_" + str(index_group))
                    nums.append(element["name"] + "_max_" + str(index_group))
                    nums.append(element["name"] + "_mean_" + str(index_group))
        if unions["single sensors"] != "null":
            [nums.append(x) for x in unions["single sensors"] if x not in approxlist]
    return nums


# Загрузка точек, определенных на этапе обучения
def points_load_online_mode(group):
    path_to_points = f"{DATA_DIR}{os.sep}{group}{os.sep}{config_json['paths']['files']['points_json']}{group}.json"
    with open(path_to_points, "r", encoding='utf8') as fh:
        points = json.load(fh)
    return points


# Загрузка таблиц потенциал-вероятность, определенных на этапе обучения
def potential_probability_load_online_mode(group):
    path_to_potential_probability = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                                    f"{config_json['paths']['files']['table_potential_probability']}{group}.csv"
    potential_probability = pd.read_csv(path_to_potential_probability)
    return potential_probability


# Вычисление потенциала и loss для онлайн режима
def potentials_analyse_online_mode(data, nums, points):
    # словарь с нормализованными значениями датчиков группы
    data_norm = {}
    for num in nums:
        data_norm[num] = data[num]

    # суммарный потенциал
    sum_p = 0
    # словарь суммарного потенциала датчиков группы
    sum_num = {n: 0 for n in nums}

    # вычисление потенциала
    for p in points:
        R = 0
        for num in nums:
            p_norm = p[num]
            delta = (p_norm - data_norm[num]) ** 2
            R += delta
            sum_num[num] += delta / N_L
        sum_p += 1 / (1 + R)
    sum_p = sum_p / ((N_L+2)*len(nums))
    return sum_p, sum_num


def online_sochi():
    freeze = False
    client = clickhouse_connect.get_client(host=config_json['paths']['database']['clickhouse']['host_ip'],
                                           username=config_json['paths']['database']['clickhouse']['username'],
                                           password=config_json['paths']['database']['clickhouse']['password'])
    try:
        while True:
            # Чтение ненормализованной и необъединенной последней строки из таблицы
            last_row = client.query_df("SELECT * from slices_play order by timestamp desc limit 1").tail(1)
            print(last_row)
            last_row.drop(columns=blacklist, inplace=True)
            # Отсечка по мощности
            if last_row[approxlist[1]][0] <= config_json['model']['N']:
                print(f'текущая мощность = {last_row[approxlist[1]][0]} <= {config_json["model"]["N"]}')
                print(f'обрабатываем без вычислений потенциал, loss, вероятность')
                # Запрос на предпоследнюю строку
                #penult_last = client.query_df("SELECT * from slices_play order by timestamp desc limit 2").tail(2)
                #last_row[approxlist[1]][0] = penult_last.iloc[1][approxlist[1]]
                #print("freeze power =", last_row[approxlist[1]][0])
                freeze = True
            else:
                freeze = False
                # Объединение, отстройка и нормализация
                row_union = union.unite_row(f"{DATA_DIR}{os.sep}{config_json['paths']['files']['json_sensors']}",
                                            last_row.to_dict())
                row_norm = normalization.normalize_multi_regress_two_powers_row(row_union,
                                                                                f"{DATA_DIR}{os.sep}{config_json['paths']['files']['coef_train_json']}",
                                                                                approxlist[1], approxlist[0])
                # Выбрасываем отстроечные параметры из словаря
                del row_norm[approxlist[0]]
                del row_norm[approxlist[1]]

            # Цикл расчета по всем группам кроме "0"
            for group in index_group:
                print("group = ", group)
                # Мощность < уставной -> морозим без вычислений потенциал, loss, вероятность
                if freeze:
                    # Запрос на предпоследнюю строку потенциала, loss и вероятности
                    penult_predict_row = client.query_df(f"SELECT * FROM potential_predict_{group} order by timestamp desc limit 1").tail(1)
                    penult_loss_row = client.query_df(f"SELECT * FROM potential_loss_{group} order by timestamp desc limit 1").tail(1)
                    if penult_predict_row.empty or penult_loss_row.empty:
                        continue
                    potential = 100.0
                    prob = 0.0
                    # potential = penult_predict_row['potential'].values[0]
                    # prob = penult_predict_row['probability'].values[0]
                    penult_loss_row.drop(columns='timestamp', inplace=True)

                    loss = penult_loss_row
                    loss = loss.to_dict('records')[0]

                    delta_tau_P[group] = 0
                    delta_tau_T[group] = 0

                    print("Замороженные значения")
                    print("Потенциал = ", potential)
                    print("loss", loss)
                    print("Вероятность = ", prob)
                else:
                    # Вычисление потенциала
                    potential, loss = potentials_analyse_online_mode(row_norm, nums[group], points[group])
                    #print("Потенциал = ", potential)

                    # Вычисление вероятности через таблицы распределений потенциал-вероятность
                    min_pot = min(potential_probability[group]['potential'], key=lambda x: abs(x - potential))
                    ind = potential_probability[group][potential_probability[group]['potential'] == min_pot].index

                    prob = potential_probability[group]['probability'].iloc[ind].values[0]
                    #print("Вероятность = ", prob)
                if prob >= config_json['model']['P_pr'] * 100:
                    print(last_row['timestamp'].values[0], prob, "ANOMALY")
                    anomaly_time = "0"
                    regress_prob[group] = []
                    index[group] = 0
                    # Критерий по достижению вероятности
                    delta_tau_P[group] += 1
                    # Критерий по достижению прогнозируемого времени
                    delta_tau_T[group] = 0
                    KrT[group] = 0
                    if (delta_tau_P[group] > (config_json['model']['delta_tau_P'] * config_json['number_of_samples']))\
                            and (not freeze):
                        KrP[group] = 1
                    else:
                        KrP[group] = 0
                else:
                    # Критерий по достижению вероятности
                    delta_tau_P[group] = 0
                    KrP[group] = 0
                    #regress_prob[group].append(prob)
                    index[group] += 1
                    # Если данных для окна хватает
                    if index[group] > regress_days:
                        # Запрос на вероятности для формирования окна регрессии
                        regress_prob_rows = client.query_df(f"SELECT * FROM potential_predict_{group} order by timestamp desc limit {regress_days}").tail(
                            regress_days)
                        if regress_prob_rows.empty or (len(regress_prob_rows) < regress_days):
                            continue
                        regress_prob[group] = regress_prob_rows['probability'].to_list()
                        x = np.array(range(index[group]-regress_days, index[group], config_json['model']['s'])).reshape(
                            (-1, 1))  # x в окне
                        # y = np.array(regress_prob[group][index[group] - regress_days:index[group]:config_json['model']['s']])  # вероятность
                        y = np.array(regress_prob[group])  # вероятность
                        model = LinearRegression().fit(x, y)
                        root = (config_json['model']['P_pr'] * 100 - model.intercept_) / model.coef_[0]
                        print("work regression")
                        if (model.coef_[0] > 0) and (((root - float(index[group])) / config_json['number_of_samples']) < 720):
                            # Корень либо внутри окна,либо правее него
                            # если внутри окна то ноль так как рост аномалии указывал на момент времени в прошлом
                            # Аномалия могла уже наступить
                            anomaly_time = 0.0 if max(0, (root - float(index[group])) / config_json['number_of_samples']) == 0 \
                                else ((root - float(index[group])) / config_json['number_of_samples'])
                            # Критерий по достижению прогнозируемого времени
                            if (datetime.timedelta(hours=anomaly_time) <=
                                    datetime.timedelta(hours=config_json['model']['T_pr'])):
                                delta_tau_T[group] += 1
                                if delta_tau_T[group] > (
                                        config_json['model']['delta_tau_T'] * config_json['number_of_samples']) \
                                        and (not freeze):
                                    KrT[group] = 1
                                else:
                                    KrT[group] = 0
                            else:
                                delta_tau_T[group] = 0
                                KrT[group] = 0
                            print(last_row['timestamp'].values[0], prob, (root - float(index[group])) / config_json['number_of_samples'])
                        else:
                            # либо вероятно падает либо до нее > 1 минуты
                            anomaly_time = "NaN"
                            KrT[group] = 0
                            delta_tau_T[group] = 0
                            regress_prob[group] = []
                            index[group] = 0
                            print(last_row['timestamp'].values[0], prob, "N/A")
                    else:
                        # Недостаточно данных для формирования окна
                        anomaly_time = "NaN"
                        KrT[group] = 0
                        delta_tau_T[group] = 0
                        print(last_row['timestamp'].values[0], prob, "Window...")
                print("Время до аномалии: ", anomaly_time)
                # Сохранение в БД таблиц loss по группам

                loss_df = [loss]
                loss_df = pd.DataFrame(loss_df)
                loss_df['timestamp'] = last_row['timestamp']

                client.insert_df(f'potential_loss_{group}', loss_df)
                # Достаем для проверки из БД добавленную строчку loss
                #new_loss_records = client.query_df(f"SELECT * FROM potential_loss_{group} order by timestamp desc limit 1").tail(1)

                predict_df = pd.DataFrame({'potential': potential, 'probability': prob,
                                           'anomaly_time': str(anomaly_time),
                                           'KrP': bool(KrP[group]), 'KrT': bool(KrT[group]),
                                           'timestamp': last_row['timestamp']})

                client.insert_df(f'potential_predict_{group}', predict_df)

                # Достаем для проверки из БД добавленную строчку predict
                new_predict_records = client.query_df(f'SELECT * FROM potential_predict_{group} order by timestamp desc limit 5').tail(5)
                print(new_predict_records)

            time.sleep(5)
    finally:
        client.close()
        print("disconnected")


def create_online_table():
    client = clickhouse_connect.get_client(host='10.23.0.177', username='default', password='asdf')
    try:
        # Цикл создания таблиц по группам
        for group in index_group:
            print("group = ", group)
            # Создание в БД таблиц loss по группам

            col_str = ''
            for col in nums[group]:
                col_str += '"' + col + '"' + ' ' + 'Float64, '
            col_str += '"' + 'timestamp' + '"' + ' ' + 'DateTime64'

            client.command(f'DROP TABLE IF EXISTS potential_loss_{group}')
            client.command(f'CREATE TABLE potential_loss_{group} ({col_str}) ENGINE = Memory')

            # Создание в БД таблицы под потенциал, вероятность и время до аномалии

            col_str = 'potential Float64, probability Float64, anomaly_time String,' \
                      'KrP Boolean, KrT Boolean, timestamp DateTime64'
            client.command(f'DROP TABLE IF EXISTS potential_predict_{group}')
            client.command(f'CREATE TABLE potential_predict_{group} ({col_str}) ENGINE = Memory')
    finally:
        client.close()
        print("disconnected")


if __name__ == '__main__':
    parser = create_parser()
    namespace = parser.parse_args()

    with open("config_SOCHI.json", 'r', encoding='utf8') as j:
        config_json = json.load(j)
    with open(f"{DATA_DIR}{os.sep}{config_json['paths']['files']['json_sensors']}", 'r', encoding='utf8') as f:
        json_dict = json.load(f)
    N_L = config_json['model']['N_l']

    blacklist = config_json['model']['blacklist_sensors']
    approxlist = config_json['model']['approx_sensors']

    # Нахождение групп и загрузка датчиков групп, точек
    index_group = [list(x.keys())[0] for x in json_dict["groups"]]
    if index_group[0] == '0':
        index_group.remove('0')
    nums = {}
    points = {}
    potential_probability = {}

    delta_tau_P = {}
    delta_tau_T = {}

    KrP = {}
    KrT = {}

    regress_prob = {}
    index = {}
    for group in index_group:
        # Добавление KKS в словарь по ключу группы
        nums[group] = kks_online_mode(approxlist, group)
        # Добавление точек в словарь по ключу группы
        points[group] = points_load_online_mode(group)
        # Добавление таблиц потенциал-вероятность в словарь по ключу группы
        potential_probability[group] = potential_probability_load_online_mode(group)

        delta_tau_P[group] = 0
        delta_tau_T[group] = 0

        KrP[group] = 0
        KrT[group] = 0

        regress_prob[group] = []
        index[group] = 0
    regress_days = config_json['model']['delta'] * config_json['number_of_samples']  # окно - период
    # regress_days = 1 * config_json['number_of_samples']  # окно - 1 минута
    # regress_days = 3  # окно - 15 секунд
    if config_json["create_online_table"] == 1:
        create_online_table()
    online_sochi()
