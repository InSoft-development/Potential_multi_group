import argparse
import os
import sys
import json
import pandas as pd

import os

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_anomaly_time", nargs=1, help="path to CSV file with anomaly date")
    parser.add_argument("path_to_loss", nargs=1, help="path to CSV file with loss")
    parser.add_argument("path_to_threshold", nargs=1, help="path to json with thresholds")
    parser.add_argument("path_to_index_sensors", nargs=1, help="path to json files which contained index|sensors")
    parser.add_argument("path_to_intervals_json", nargs=1, help="path to saving json file with anomalies intervals")
    return parser


def interval_group(group):
    print("group", group)
    df_anomaly = pd.read_csv(path_to_anomaly_time, index_col=[0])

    df_anomaly['t'] = pd.to_datetime(df_anomaly['t'], format="%Y-%m-%d %H:%M:%S")
    print(df_anomaly)

    df_loss = pd.read_csv(path_to_loss)

    j = 0
    jT = 0
    d1 = df_anomaly["t"][0]
    d1T = df_anomaly['t'][0]
    dict_list = []
    interval_begin_index = 0
    interval_begin_indexT = 0

    # New approach
    for index, row in df_anomaly.iterrows():
        if row['KrP']:
            if j == 0:
                d1 = df_anomaly['t'][index]
                interval_begin_index = index
            j += 1
        if (not row['KrP']) and (j != 0):
            d2 = df_anomaly["t"][index]
            print("KrP")
            print(d1, " - ", d2, "time in hours", j * 5 / 60)
            #input("Press Enter to continue...")
            #top_P = max_index(group, df_anomaly[interval_begin_index:index])
            top_P = mean_index(df_loss[interval_begin_index:index])
            dictionary = {
                "time": (str(d1), str(d2)),
                "len": index - interval_begin_index,
                "index": (interval_begin_index, index),
                "top_sensors": top_P
            }
            dict_list.append(dictionary)
            j = 0
        if row['KrT']:
            if jT == 0:
                d1T = df_anomaly['t'][index]
                interval_begin_indexT = index
            jT += 1
        if (not row['KrT']) and (jT != 0):
            d2T = df_anomaly["t"][index]
            print("KrT")
            print(d1T, " - ", d2T, "time in hours", jT * 5 / 60)
            #input("Press Enter to continue...")
            #top_T = max_index(group, df_anomaly[interval_begin_indexT:index])
            top_T = mean_index(df_loss[interval_begin_index:index])
            dictionary = {
                "time": (str(d1T), str(d2T)),
                "len": index - interval_begin_indexT,
                "index": (interval_begin_indexT, index),
                "top_sensors": top_T
            }
            dict_list.append(dictionary)
            jT = 0
    try:
        os.mkdir("json_interval")
    except Exception as e:
        print(e)
        print('Directory exist')
    with open(path_to_intervals_json, 'w', encoding='utf8') as f:
        json.dump(dict_list, f, ensure_ascii=False, indent=4)


def mean_index(data, top_count=3):
    mean_loss = data.mean().sort_values(ascending=False).index[:top_count].to_list()
    return mean_loss


def max_index(group, data):
    with open(path_to_index_sensors + str(group) + ".json", 'r', encoding='utf8') as j:
        index_sensors_json = json.load(j)
    print(index_sensors_json)

    sum_index = {}
    for i in data["index0"]:
        if i in sum_index:
            sum_index[i] += 3
        else:
            sum_index[i] = 3

    for i in data["index1"]:
        if i in sum_index:
            sum_index[i] += 2
        else:
            sum_index[i] = 2

    for i in data["index2"]:
        if i in sum_index:
            sum_index[i] += 1
        else:
            sum_index[i] = 1

    for i in data["index3"]:
        if i in sum_index:
            sum_index[i] += 1
        else:
            sum_index[i] = 1

    for i in data["index4"]:
        if i in sum_index:
            sum_index[i] += 1
        else:
            sum_index[i] = 1
    print(sum_index)
    sorted_i = list(dict(sorted(sum_index.items(), key=lambda item: item[1], reverse=True)).keys())
    top = []
    print(sorted_i)
    for i in sorted_i[0:3]:
        print(i)
        print(index_sensors_json[str(int(i))])
        top.append(index_sensors_json[str(int(i))])
    return top


if __name__ == '__main__':
    # Заполнение коэффициентов json из всего dataframe
    parser = create_parser()
    with open("config_SOCHI.json", 'r', encoding='utf8') as j:
        config_json = json.load(j)
    if len(sys.argv) == 1:
        print("config SOCHI")
        print("Enter 0 for choose anomaly_time_prob frame\nEnter 1 for choose anomaly_time_intercept frame")
        input_anomaly_file = input()
        with open(config_json['paths']['files']['json_sensors'], 'r', encoding='utf8') as f:
            json_dict = json.load(f)

        index_group = [list(x.keys())[0] for x in json_dict["groups"]]
        if index_group[0] == '0':
            index_group.remove('0')
        for group in index_group:
            if int(input_anomaly_file) == 0:
                path_to_anomaly_time = str(group) + os.sep + config_json['paths']['files']['anomaly_time_prob']
            if int(input_anomaly_file) == 1:
                path_to_anomaly_time = str(group) + os.sep + config_json['paths']['files']['anomaly_time_intercept']
            path_to_loss = str(group) + os.sep + config_json['paths']['files']['loss_csv']
            path_to_threshold_json = str(group) + os.sep + config_json['paths']['files']['threshold_json']
            path_to_index_sensors = config_json['paths']['files']['index_sensors_json'] + str(group)+".json"
            path_to_intervals_json = config_json['paths']['files']['intervals_json'] + str(group) + ".json"
            interval_group(str(group))
    else:
        print("command's line arguments")
        namespace = parser.parse_args()
        path_to_anomaly_time = namespace.path_to_anomaly_time[0]
        path_to_loss = namespace.path_to_loss[0]
        path_to_threshold_json = namespace.path_to_threshold[0]
        path_to_index_sensors = namespace.path_to_index_sensors[0]
        path_to_intervals_json = namespace.path_to_intervals_json[0]
        for g in range(1, config_json['count_of_groups']+1):
            interval_group(str(g))