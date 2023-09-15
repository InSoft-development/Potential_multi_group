import argparse
import json
import os
import sqlite3

import pandas as pd
import time

DATA_DIR = f'Data'

nums = []
points = {}


def create_parser():
    parser = argparse.ArgumentParser(description="calculate potentials by groups")
    parser.add_argument("-v", "--version", action="version", help="print version", version="1.0.1")
    return parser


def kks_load():
    global nums
    nums = []
    with open(file_json, 'r', encoding='utf8') as f:
        json_dict = json.load(f)

    index_group = [list(x.keys())[0] for x in json_dict["groups"]]
    try:
        index_group = index_group.index(str(group))
    except ValueError:
        print("Группы " + str(group) + " не существует")
        return False

    for unions in json_dict["groups"][index_group].values():
        if unions["unions"] != "null":
            for union_val in unions["unions"]:
                for element in union_val.values():
                    # Добавляем тэги union датчиков
                    nums.append(element["name"] + "_min_" + str(index_group))
                    nums.append(element["name"] + "_max_" + str(index_group))
                    nums.append(element["name"] + "_mean_" + str(index_group))
        if unions["single sensors"] != "null":
            [nums.append(x) for x in unions["single sensors"] if x not in power]
    #print("Sensors in group\n", nums)
    return True


def points_load():
    with open(path_to_points, "r", encoding='utf8') as fh:
        global points
        points = json.load(fh)
    #print("points\n", points)


def potentials_analyse(data):
    # словарь с нормализованными значениями датчиков группы
    data_norm = {}
    for num in nums:
        data_norm[num] = data[num]

    # суммарный потенциал
    sum_p = 0
    # словарь суммарного потенциала датчиков группы
    sum_num = {n: 0 for n in nums}
    points_length = config_json['model']['N_l']

    # вычисление потенциала
    # Засечь
    start = time.time()
    for p in points:
        R = 0
        for num in nums:
            p_norm = p[num]
            delta = (p_norm - data_norm[num]) ** 2
            R += delta
            sum_num[num] += delta / points_length
        sum_p += 1 / (1 + R)
    # sum_p = sum_p / len(nums)
    sum_p = sum_p / ((points_length+2)*len(nums))
    end = time.time() - start
    global t_sum
    t_sum += end
    #
    #sum_num = sum_num / len(points)
    #print(sum_num)
    #sorted_indexes = sorted(sum_num, key=sum_num.get, reverse=True)
    # print(sorted_indexes)
    # print(sorted_indexes[0:5])
    #sorted_numbers = [list(nums).index(i) for i in sorted_indexes[0:5]]
    #return sum_p, sorted_numbers, sum_num
    return sum_p, sum_num


def analyse_loop_month_one_powers(file_name, file_power):
    print("Data from file", file_name)
    # DataFrame с необъединенными и ненормализованными данными
    con = sqlite3.connect(file_name)
    df = pd.read_sql_query("SELECT * from data", con)
    df_power = pd.read_csv(file_power)
    #print(df.head())

    anomaly = []
    t = []
    flag = True
    rotor = []
    # Аномальность
    anomaly_index = []
    # подготовка массивов с индексами 5-ки "наибольших" центров
    for i in range(0, 5):
        anomaly_index.append([])
    # Количество строк DataFrame
    N = len(df.index)
    #print("N = ", N)

    for index, row in df.iterrows():
        temp_row = row.to_dict()
        # Суммарный потенциал и индексы 5-ти "наибольших" центров
        #
        a, loss = potentials_analyse(temp_row)
        #
        #print("a = ", a, "s = ", s)
        # Значение мощности
        r = df_power.iloc[index][power[0]]

        if flag:
            if a > 100:
                #print(row["timestamp"])
                flag = False

        anomaly.append(a)  # добавление в массив значения суммарного потенциала

        rotor.append(r)  # добавление в массив значения датчика мощности

        # Проход по "наибольшим" центрам
        # for i in range(0, len(s) if len(s) < 5 else 5):
        #     anomaly_index[i].append(s[i])  # добавление в массив индексов "наибольших" центров

        # добавляем дату
        t.append(row["timestamp"])
        '''if index == (N-1):
            df = pd.DataFrame({'timestamp': t,
                               'potential': anomaly,
                               'N': rotor})
            for i in range(0, len(s) if len(s) < 5 else 5):
                df['index'+str(i)] = anomaly_index[i]
            df.to_csv(group + "/" + config_json['paths']['files']['potentials_csv'], index=False)'''
        print(int(index * 100 / N), "%")
    df = pd.DataFrame({'timestamp': t,
                       'potential': anomaly,
                       'N': rotor})
    # for i in range(0, len(s) if len(s) < 5 else 5):
    #     df['index' + str(i)] = anomaly_index[i]
    df.to_csv(f"{DATA_DIR}{os.sep}{group}{os.sep}{config_json['paths']['files']['potentials_csv']}{group}.csv", index=False)

    df_loss = pd.DataFrame(data=loss)
    df_loss['timestamp'] = t
    df_loss.to_csv(f"{DATA_DIR}{os.sep}{group}{os.sep}{config_json['paths']['files']['loss_csv']}{group}.csv", index=False)


def analyse_loop_month_two_powers(file_name, file_power):
    print("Data from file", file_name)
    # DataFrame с необъединенными и ненормализованными данными
    con = sqlite3.connect(file_name)
    df = pd.read_sql_query("SELECT * from data", con)
    df_power = pd.read_csv(file_power)
    #print(df.head())

    anomaly = []
    t = []
    flag = True
    rotor_1 = []
    rotor_2 = []
    loss_list = []
    # Аномальность
    anomaly_index = []
    # подготовка массивов с индексами 5-ки "наибольших" центров
    for i in range(0, 5):
        anomaly_index.append([])
    # Количество строк DataFrame
    N = len(df.index)
    #print("N = ", N)

    for index, row in df.iterrows():
        temp_row = row.to_dict()
        # Суммарный потенциал и индексы 5-ти "наибольших" центров
        a, loss = potentials_analyse(temp_row)
        #print("a = ", a, "s = ", s)
        # Значение мощности
        r_1 = df_power.iloc[index][power[0]]
        r_2 = df_power.iloc[index][power[1]]
        loss_list.append(loss)

        if flag:
            if a > 100:
                #print(row["timestamp"])
                flag = False

        anomaly.append(a)  # добавление в массив значения суммарного потенциала

        rotor_1.append(r_1)  # добавление в массив значения датчика мощности
        rotor_2.append(r_2)

        # Проход по "наибольшим" центрам
        # for i in range(0, len(s) if len(s) < 5 else 5):
        #     anomaly_index[i].append(s[i])  # добавление в массив индексов "наибольших" центров

        # добавляем дату
        t.append(row["timestamp"])
        '''if index == (N-1):
            df = pd.DataFrame({'timestamp': t,
                               'potential': anomaly,
                               'T': rotor_1,
                               'N': rotor_2})
            for i in range(0, len(s) if len(s) < 5 else 5):
                df['index'+str(i)] = anomaly_index[i]
            df.to_csv(group + "/" + config_json['paths']['files']['potentials_csv'], index=False)'''
        print(int(index * 100 / N), "%")
    df = pd.DataFrame({'timestamp': t,
                       'potential': anomaly,
                       'T': rotor_1,
                       'N': rotor_2})
    # for i in range(0, len(s) if len(s) < 5 else 5):
    #     df['index' + str(i)] = anomaly_index[i]
    df.to_csv(f"{DATA_DIR}{os.sep}{group}{os.sep}{config_json['paths']['files']['potentials_csv']}{group}.csv", index=False)

    df_loss = pd.DataFrame(data=loss_list)
    df_loss['timestamp'] = t
    df_loss.to_csv(f"{DATA_DIR}{os.sep}{group}{os.sep}{config_json['paths']['files']['loss_csv']}{group}.csv", index=False)


def period_analyse():
    if not kks_load():
        return
    else:
        points_load()
        print("analyse")
        if len(config_json['model']['approx_sensors']) == 1:
            analyse_loop_month_one_powers(row_data, row_data_with_power)
        else:
            analyse_loop_month_two_powers(row_data, row_data_with_power)


# merge loss со срезами для восполнения пропущенных значений по мощности
def loss_freeze_merge(loss_path, slices_path):
    loss = pd.read_csv(loss_path)
    slice_csv = pd.read_csv(slices_path)
    time_df = slice_csv['timestamp']
    loss = pd.merge(time_df, loss, how='left', on='timestamp')
    loss.fillna(method='ffill', inplace=True)
    loss.fillna(value={"target_value": 0}, inplace=True)
    loss.to_csv(f"{DATA_DIR}{os.sep}{group}{os.sep}{config_json['paths']['files']['loss_csv']}{group}.csv", index=False)


if __name__ == '__main__':
    parser = create_parser()
    namespace = parser.parse_args()
    with open("config_SOCHI.json", 'r', encoding='utf8') as j:
        config_json = json.load(j)
    print("config SOCHI")
    file_json = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['json_sensors']}"
    power = config_json['model']['approx_sensors']
    row_data = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['sqlite_norm']}"
    row_data_with_power = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['csv_truncate_by_power']}"
    path_to_csv = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['original_csv']}"
    with open(file_json, 'r', encoding='utf8') as f:
        json_dict = json.load(f)

    index_group = [list(x.keys())[0] for x in json_dict["groups"]]
    if index_group[0] == '0':
        index_group.remove('0')
    print(index_group)
    for group in index_group:
        path_to_points = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                         f"{config_json['paths']['files']['points_json']}{str(group)}.json"
        path_to_loss = f"{DATA_DIR}{os.sep}{group}{os.sep}{config_json['paths']['files']['loss_csv']}{group}.csv"
        t_sum = 0
        start_group = time.time()
        period_analyse()
        t_all_group = time.time() - start_group
        print(f'Суммарное время работы математики группы {group} = {t_sum}')
        print(f'Время отработки группы {group} = {t_all_group}')
        loss_freeze_merge(path_to_loss, path_to_csv)
