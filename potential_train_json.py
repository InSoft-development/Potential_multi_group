import argparse
import sys
import json
import sqlite3

import pandas as pd


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_json", nargs=1, help="json file with unions and sensors")
    parser.add_argument("row_data", nargs=1, help="name of input sqlite with rows of data")
    parser.add_argument("N", nargs=1, help="counts of points")
    parser.add_argument("path_to_save", nargs=1, help="path to save points json (format e.g. points_)")
    parser.add_argument("-drop_sensors", nargs='+', help="sensor of powers which should be dropped", required=True)
    return parser


if __name__ == '__main__':
    parser = create_parser()
    with open("config_SOCHI.json", 'r', encoding='utf8') as j:
        config_json = json.load(j)
    if len(sys.argv) == 1:
        print("config SOCHI_generator")
        file_name = config_json['paths']['files']['sqlite_norm']
        N = config_json['model']['N_l']
        path_json = config_json['paths']['files']['json_sensors']
        drop_sensors = config_json['model']['approx_sensors']
        path_to_save = config_json['paths']['files']['points_json']
    else:
        print("command's line arguments")
        namespace = parser.parse_args()
        file_name = str(namespace.row_data[0])
        N = int(namespace.N[0])
        path_json = str(namespace.file_json[0])
        drop_sensors = namespace.drop_sensors
        path_to_save = str(namespace.path_to_save[0])

    # DataFrame с нормализованными данными
    con = sqlite3.connect(file_name)
    df = pd.read_sql_query("SELECT * from data", con)
    print(df.head())

    with open(path_json, 'r', encoding='utf8') as f:
        json_dict = json.load(f)

    # Добавление датчиков из групп
    for group in json_dict["groups"]:
        nums = []
        for unions in group.values():
            if unions["unions"] != "null":
                for union_val in unions["unions"]:
                    for element in union_val.values():
                        # Добавляем тэги union датчиков
                        nums.append(element["name"] + "_min_" + list(group.keys())[0])
                        nums.append(element["name"] + "_max_" + list(group.keys())[0])
                        nums.append(element["name"] + "_mean_" + list(group.keys())[0])
            if unions["single sensors"] != "null":
                [nums.append(str(x)) for x in unions["single sensors"] if x not in drop_sensors]

        print(group.keys(), nums)

        # Убираем датчики, которые не входят в группу
        #df_group = df[nums]
        df_group = df

        # Словарь индексов с max и min значениями
        points_max = {}
        points_min = {}

        # Проход по каждому датчику в группе
        for num in nums:
            print(num)

            points_max[num] = df_group[num].idxmax()
            points_min[num] = df_group[num].idxmin()

        for num in nums:
            print(num, points_max[num], points_min[num])

        # Формирование точек
        points = []
        for num in nums:
            temp_norm_row_min = df_group.iloc[points_min[num]].to_dict()
            temp_norm_row_max = df_group.iloc[points_max[num]].to_dict()

            points.append(temp_norm_row_min)
            points.append(temp_norm_row_max)
            if abs(points_max[num] - points_min[num]) < N:
                print("ne pomeshaetsa, budut vse")
                # Добор точек
                for i in range(points_max[num], points_min[num]):
                    temp_norm_row_between = df_group.iloc[i].to_dict()
                    points.append(temp_norm_row_between)
            else:
                n1 = min(points_max[num], points_min[num])
                n2 = max(points_max[num], points_min[num])
                print("pomeshaetsa, budut", N)
                # N точек
                for i in range(n1, n2, (n2 - n1) // N):
                    temp_norm_row_between = df_group.iloc[i].to_dict()
                    points.append(temp_norm_row_between)
            print(len(points))
        print("length:", len(points))

        # Сохранение точек в json
        with open(path_to_save+str(list(group.keys())[0])+".json", "w", encoding="utf8") as write_file:
            json.dump(points, write_file, indent=4, ensure_ascii=False)
