import argparse
import sys
import json
import os
import sqlite3

import pandas as pd
import time

nums = []
points = {}

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_json", nargs=1, help="json file with unions and sensors")
    parser.add_argument("row_data", nargs=1, help="path of input sqlite file with normalized rows of data")
    parser.add_argument("row_data_with_power", nargs=1, help="path of input excel with unnormalized rows of data")
    parser.add_argument("group", nargs=1, help="number of sensor's group")
    parser.add_argument("path_to_points", nargs=1, help="path to files which contained points")
    parser.add_argument("path_to_index_sensors", nargs=1, help="path to json files which contained index|sensors->nums")
    parser.add_argument("-power", nargs='+', help="sensor of power: N should be typed in the end", required=True)
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
    index_sensors_json = {}
    for i in range(0, len(nums)):
        index_sensors_json[str(i)] = nums[i]
    with open(path_to_index_sensors, 'w', encoding='utf8') as f:
        json.dump(index_sensors_json, f, ensure_ascii=False, indent=4)
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
    sorted_indexes = sorted(sum_num, key=sum_num.get, reverse=True)
    # print(sorted_indexes)
    # print(sorted_indexes[0:5])
    sorted_numbers = [list(nums).index(i) for i in sorted_indexes[0:5]]
    return sum_p, sorted_numbers, sum_num


def analyse_loop_month_one_powers(file_name, file_power):
    print("Data from file", file_name)
    # DataFrame с необъединенными и ненормализованными данными
    con = sqlite3.connect(file_name)
    df = pd.read_sql_query("SELECT * from data", con)
    df_power = pd.read_excel(file_power, sheet_name="slices")
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

    if not os.path.exists(str(group)):
        os.mkdir(str(group))

    for index, row in df.iterrows():
        temp_row = row.to_dict()
        # Суммарный потенциал и индексы 5-ти "наибольших" центров
        #
        a, s, loss = potentials_analyse(temp_row)
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
        for i in range(0, len(s) if len(s) < 5 else 5):
            anomaly_index[i].append(s[i])  # добавление в массив индексов "наибольших" центров

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
    for i in range(0, len(s) if len(s) < 5 else 5):
        df['index' + str(i)] = anomaly_index[i]
    df.to_csv(str(group) + "/" + config_json['paths']['files']['potentials_csv'], index=False)

    df_loss = pd.DataFrame(data=loss)
    df_loss['timestamp'] = t
    df_loss.to_csv(str(group) + "/" + config_json['paths']['files']['loss_csv'], index=False)


def analyse_loop_month_two_powers(file_name, file_power):
    print("Data from file", file_name)
    # DataFrame с необъединенными и ненормализованными данными
    con = sqlite3.connect(file_name)
    df = pd.read_sql_query("SELECT * from data", con)
    df_power = pd.read_excel(file_power, sheet_name="slices")
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

    if not os.path.exists(str(group)):
        os.mkdir(str(group))

    for index, row in df.iterrows():
        temp_row = row.to_dict()
        # Суммарный потенциал и индексы 5-ти "наибольших" центров
        a, s, loss = potentials_analyse(temp_row)
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
        for i in range(0, len(s) if len(s) < 5 else 5):
            anomaly_index[i].append(s[i])  # добавление в массив индексов "наибольших" центров

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
    for i in range(0, len(s) if len(s) < 5 else 5):
        df['index' + str(i)] = anomaly_index[i]
    df.to_csv(str(group) + "/" + config_json['paths']['files']['potentials_csv'], index=False)

    df_loss = pd.DataFrame(data=loss_list)
    df_loss['timestamp'] = t
    df_loss.to_csv(str(group) + "/" + config_json['paths']['files']['loss_csv'], index=False)


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


if __name__ == '__main__':
    parser = create_parser()
    with open("config_SOCHI.json", 'r', encoding='utf8') as j:
        config_json = json.load(j)
    if len(sys.argv) == 1:
        print("config SOCHI")
        file_json = config_json['paths']['files']['json_sensors']
        power = config_json['model']['approx_sensors']
        row_data = config_json['paths']['files']['sqlite_norm']
        row_data_with_power = config_json['paths']['files']['excel_df']
        with open(config_json['paths']['files']['json_sensors'], 'r', encoding='utf8') as f:
            json_dict = json.load(f)

        index_group = [list(x.keys())[0] for x in json_dict["groups"]]
        if index_group[0] == '0':
            index_group.remove('0')
        print(index_group)
        for group in index_group:
            path_to_points = config_json['paths']['files']['points_json'] + str(group) + ".json"
            path_to_index_sensors = config_json['paths']['files']['index_sensors_json'] + str(group) + ".json"
            t_sum = 0
            start_group = time.time()
            period_analyse()
            t_all_group = time.time() - start_group
            print(f'Суммарное время работы математики группы {group} = {t_sum}')
            print(f'Время отработки группы {group} = {t_all_group}')
    else:
        print("command's line arguments")
        namespace = parser.parse_args()
        file_json = namespace.file_json[0]
        group = namespace.group[0]
        power = namespace.power
        path_to_points = namespace.path_to_points[0]
        path_to_index_sensors = namespace.path_to_index_sensors[0]
        row_data = namespace.row_data[0]
        row_data_with_power = namespace.row_data_with_power[0]
        period_analyse()