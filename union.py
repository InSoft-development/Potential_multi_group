import pandas as pd
import numpy as np
import json
import os
import sqlite3
import clickhouse_connect


# Функция сериализует и записывает json c выделенными union
def json_build(source, config, path_to_save, sheet='SOCHI'):
    if source == "clickhouse":
        client = clickhouse_connect.get_client(host='10.23.0.177', username='default', password='asdf')
        data_csv = client.query_df(f"{config['paths']['database']['clickhouse']['original_kks_query']}")
        data_csv.rename(columns={'kks': 0, 'name': 1, 'group': 2}, inplace=True)
        data_csv[2] = data_csv[2].astype('int64')
        data_csv[2] = data_csv[2].astype('string')
        data_group = client.query_df(f"{config['paths']['database']['clickhouse']['original_group_query']}")
        client.close()
    elif source == "sqlite":
        client = sqlite3.connect(f"{config['paths']['database']['sqlite']['original_kks']}")
        data_csv = pd.read_sql_query(f"{config['paths']['database']['sqlite']['original_kks_query']}", client)
        data_csv.rename(columns={'kks': 0, 'name': 1, 'group': 2}, inplace=True)
        data_csv[2] = data_csv[2].astype('int64')
        data_csv[2] = data_csv[2].astype('string')
        client.close()
        client = sqlite3.connect(f"{config['paths']['database']['sqlite']['original_group']}")
        data_group = pd.read_sql_query(f"{config['paths']['database']['sqlite']['original_group_query']}", client)
        client.close()
    else:
        DATA_DIR = f'Data'
        path_to_original_kks = f"{DATA_DIR}{os.sep}{config['paths']['files']['original_kks']}"
        path_to_original_group_csv = f"{DATA_DIR}{os.sep}{config['paths']['files']['original_group_csv']}"
        data_csv = pd.read_csv(path_to_original_kks, delimiter=';', header=None)
        data_csv[2] = data_csv[2].astype('string')
        data_group = pd.read_csv(path_to_original_group_csv, delimiter=',')
    print(data_group)

    if 3 not in data_csv.columns.to_list():
        data_csv[3] = np.NaN
    json_dict = {
        "station": sheet,
        "groups": [],
        "no groups": []
    }

    # Добавление групп
    unique_group = data_csv[2].unique()
    temp_unique = []
    for uniqum in unique_group:
        if not pd.isna(uniqum):
            s = uniqum.split(",")
            for t in s:
                if t not in temp_unique:
                    temp_unique.append(t)
    # Удаление nan
    unique_group = temp_unique
    unique_group.sort(key=lambda k: int(k))
    for group in unique_group:
        group_dict = {
            str(int(group)): {
                "name": data_group.iloc[int(group)]['name'],
                "unions": "null",
                "single sensors": "null"
            }
        }
        temp_rows = data_csv[data_csv[2].str.contains(group, regex=False) == True]
        correct_temp_index = []
        for index, row in temp_rows.iterrows():
            split_array = str(row[2]).split(sep=',')
            split_array = [int(i) for i in split_array]
            if int(group) in split_array:
                correct_temp_index.append(index)
        temp_rows = temp_rows.loc[correct_temp_index]
        unique_unions = temp_rows[3].unique()
        marked_union = []
        for union in unique_unions:
            # Непомеченные union датчики
            if pd.isna(union):
                single_sensors = temp_rows.loc[(temp_rows[3].isnull()), [0]]
                single_sensors = single_sensors[0].to_list()
                single_sensors = list(map(str, single_sensors))
                group_dict[str(int(group))]["single sensors"] = single_sensors
            else:
                # Помеченные union датчики
                temp_union = union.split(',')
                print("union=", union)
                for temp in temp_union:
                    print(temp)
                    if ("("+group+")" in temp) and (temp not in marked_union):
                        print(temp)
                        union_rows = temp_rows.loc[(temp_rows[3].str.contains(temp, regex=False) == True)]
                        print(union_rows)
                        union_dict = {
                            str(int(temp[:temp.find('(')])): {
                                "name": union_rows["Name"].iloc[0],
                                "sensors": "null",
                            }
                        }
                        print(union_dict)
                        union_sensors = union_rows[0].to_list()
                        if group_dict[str(int(group))]["unions"] == "null":
                            group_dict[str(int(group))]["unions"] = []
                        union_dict[str(int(temp[:temp.find('(')]))]["sensors"] = union_sensors
                        group_dict[str(int(group))]["unions"].append(union_dict)
                        marked_union.append(temp)
        json_dict["groups"].append(group_dict)

    # Непомеченные группы
    no_group_sensors = data_csv.loc[(pd.isna(data_csv[2]))][0].to_list()
    json_dict["no groups"] = str(no_group_sensors)
    print(json_dict)

    # Запись json дампа
    with open(path_to_save, 'w', encoding='utf8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)


# создание соединения с sqlite
def connect_sqlite(db_file):
    try:
        connection = sqlite3.connect(db_file)
        return connection
    except sqlite3.Error as error:
        print("Error has been occurred in connection to sqlite", error)


# Функция считывает json и выполняет объединение датчиков в dataframe
def unite(dict, df_csv_path):
    df = pd.read_csv(df_csv_path)
    #df.rename(columns={"External Tags": "timestamp"}, inplace=True)
    time = df["timestamp"]
    data = df.drop(columns=["timestamp"])

    #print(data)
    # Объединение датчиков в группы в dataframe
    delete_sensors = []
    for group in dict["groups"]:
        for unions in group.values():
            if unions["unions"] != "null":
                for union in unions["unions"]:
                    for element in union.values():
                        input_after_me = int(np.where(data.columns.values == element["sensors"][-1])[0][0])
                        union_min = data[element["sensors"]].min(axis=1)
                        union_max = data[element["sensors"]].max(axis=1)
                        union_mean = data[element["sensors"]].mean(axis=1)
                        data.insert(input_after_me + 1, element["name"] + "_min_" + list(group.keys())[0], union_min)
                        data.insert(input_after_me + 2, element["name"] + "_max_" + list(group.keys())[0], union_max)
                        data.insert(input_after_me + 3, element["name"] + "_mean_" + list(group.keys())[0], union_mean)
                        for d in element["sensors"]:
                            if d not in delete_sensors:
                                delete_sensors.append(d)
    for d in delete_sensors:
        data.drop(columns=d, inplace=True)
    data.insert(0, "timestamp", time)
    return data


# Функция реализует union объединение датчиков
def unite_dictionary(replacement_dict, unions_dictionary):
    temp_df = pd.DataFrame(replacement_dict, index=[0])
    delete_sensors = []
    for group in unions_dictionary["groups"]:
        for unions in group.values():
            if unions["unions"] != "null":
                for union in unions["unions"]:
                    for element in union.values():
                        input_after_me = int(np.where(temp_df.columns.values == element["sensors"][-1])[0][0])
                        union_min = temp_df[element["sensors"]].min(axis=1)
                        union_max = temp_df[element["sensors"]].max(axis=1)
                        union_mean = temp_df[element["sensors"]].mean(axis=1)
                        temp_df.insert(input_after_me + 1, element["name"] + "_min_" + list(group.keys())[0], union_min)
                        temp_df.insert(input_after_me + 2, element["name"] + "_max_" + list(group.keys())[0], union_max)
                        temp_df.insert(input_after_me + 3, element["name"] + "_mean_" + list(group.keys())[0], union_mean)
                        for d in element["sensors"]:
                            if d not in delete_sensors:
                                delete_sensors.append(d)
    for d in delete_sensors:
        temp_df.drop(columns=d, inplace=True)
    return temp_df.to_dict(orient="records")


# Функция выполняет union объединение для одной строки данных
def unite_row(station_path, dict):
    """
    :param station_path: path to station's file
    :param dict: dict contained unions
    :return: dictionary with unions of sensors
    """
    try:
        dict_unite = unite_dictionary(dict, unite_row.unions)
    except AttributeError:
        with open(station_path, 'r', encoding='utf8') as f:
            unions_json = json.load(f)
        unite_row.unions = unions_json
        dict_unite = unite_dictionary(dict, unite_row.unions)
    return dict_unite[0]
