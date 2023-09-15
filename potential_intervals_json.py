import argparse
import json
import pandas as pd

import os


DATA_DIR = f'Data'


def create_parser():
    parser = argparse.ArgumentParser(description="select intervals")
    parser.add_argument("-m", "--method", type=str,
                        help="indicate the selected method for calculating time to anomaly: type probability "
                             "or intercept")
    parser.add_argument("-v", "--version", action="version", help="print version", version="1.0.1")
    return parser


def interval_group(group):
    print("group", group)
    df_anomaly = pd.read_csv(path_to_anomaly_time)

    df_anomaly['timestamp'] = pd.to_datetime(df_anomaly['timestamp'], format="%Y-%m-%d %H:%M:%S")
    print(df_anomaly)

    df_loss = pd.read_csv(path_to_loss)

    j = 0
    jT = 0
    d1 = df_anomaly["timestamp"][0]
    d1T = df_anomaly['timestamp'][0]
    dict_list = []
    interval_begin_index = 0
    interval_begin_indexT = 0

    # New approach
    for index, row in df_anomaly.iterrows():
        if row['KrP']:
            if j == 0:
                d1 = df_anomaly['timestamp'][index]
                interval_begin_index = index
            j += 1
        if (not row['KrP']) and (j != 0):
            d2 = df_anomaly["timestamp"][index]
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
                d1T = df_anomaly['timestamp'][index]
                interval_begin_indexT = index
            jT += 1
        if (not row['KrT']) and (jT != 0):
            d2T = df_anomaly["timestamp"][index]
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
        os.mkdir(f"{DATA_DIR}{os.sep}json_interval{os.sep}")
    except Exception as e:
        print(e)
        print('Directory exist')
    with open(path_to_intervals_json, 'w', encoding='utf8') as f:
        json.dump(dict_list, f, ensure_ascii=False, indent=4)


# Определение датчиков, внесших max вклад в аномалию через loss
def mean_index(data, top_count=3):
    mean_loss = data.mean().sort_values(ascending=False).index[:top_count].to_list()
    return mean_loss


# Игра в бисер - определение датчиков, внесших max вклада в аномалию
# def max_index(group, data):
#     with open(path_to_index_sensors + str(group) + ".json", 'r', encoding='utf8') as j:
#         index_sensors_json = json.load(j)
#     print(index_sensors_json)
#
#     sum_index = {}
#     for i in data["index0"]:
#         if i in sum_index:
#             sum_index[i] += 3
#         else:
#             sum_index[i] = 3
#
#     for i in data["index1"]:
#         if i in sum_index:
#             sum_index[i] += 2
#         else:
#             sum_index[i] = 2
#
#     for i in data["index2"]:
#         if i in sum_index:
#             sum_index[i] += 1
#         else:
#             sum_index[i] = 1
#
#     for i in data["index3"]:
#         if i in sum_index:
#             sum_index[i] += 1
#         else:
#             sum_index[i] = 1
#
#     for i in data["index4"]:
#         if i in sum_index:
#             sum_index[i] += 1
#         else:
#             sum_index[i] = 1
#     print(sum_index)
#     sorted_i = list(dict(sorted(sum_index.items(), key=lambda item: item[1], reverse=True)).keys())
#     top = []
#     print(sorted_i)
#     for i in sorted_i[0:3]:
#         print(i)
#         print(index_sensors_json[str(int(i))])
#         top.append(index_sensors_json[str(int(i))])
#     return top


if __name__ == '__main__':
    # Заполнение коэффициентов json из всего dataframe
    parser = create_parser()
    namespace = parser.parse_args()
    with open("config_SOCHI.json", 'r', encoding='utf8') as j:
        config_json = json.load(j)
    if namespace.method:
        if (namespace.method != "probability") and (namespace.method != "intercept"):
            print("Please choose method: -m {probability, intercept}")
            exit(0)
        print("config SOCHI")
        # print("Enter 0 for choose anomaly_time_prob frame\nEnter 1 for choose anomaly_time_intercept frame")
        # input_anomaly_file = input()
        with open(f"{DATA_DIR}{os.sep}{config_json['paths']['files']['json_sensors']}", 'r', encoding='utf8') as f:
            json_dict = json.load(f)

        index_group = [list(x.keys())[0] for x in json_dict["groups"]]
        if index_group[0] == '0':
            index_group.remove('0')
        for group in index_group:
            if namespace.method == "probability":
                path_to_anomaly_time = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                                       f"{config_json['paths']['files']['anomaly_time_prob']}{group}.csv"
            if namespace.method == "intercept":
                path_to_anomaly_time = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                                       f"{config_json['paths']['files']['anomaly_time_intercept']}{group}.csv"
            path_to_loss = f"{DATA_DIR}{os.sep}{group}{os.sep}{config_json['paths']['files']['loss_csv']}{group}.csv"
            path_to_intervals_json = f"{DATA_DIR}{os.sep}json_interval{os.sep}" \
                                     f"{config_json['paths']['files']['intervals_json']}{group}.json"
            interval_group(str(group))
