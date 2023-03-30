import pandas as pd
import numpy as np
import json
import sqlite3


# Функция сериализует и записывает json c выделенными union
def json_build(file_excel, sheet, path_to_save):
    data_excel = pd.read_excel(file_excel, sheet_name=sheet,
                               converters={"Groups": str})
    json_dict = {
        "station": sheet,
        "groups": [],
        "no groups": []
    }

    # Добавление групп
    unique_group = data_excel["Groups"].unique()
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
                "unions": "null",
                "single sensors": "null"
            }
        }
        temp_rows = data_excel[data_excel["Groups"].str.contains(group, regex=False) == True]
        correct_temp_index = []
        for index, row in temp_rows.iterrows():
            split_array = str(row["Groups"]).split(sep=',')
            split_array = [int(i) for i in split_array]
            if int(group) in split_array:
                correct_temp_index.append(index)
        temp_rows = temp_rows.loc[correct_temp_index]
        unique_unions = temp_rows["Unions"].unique()
        marked_union = []
        for union in unique_unions:
            # Непомеченные union датчики
            if pd.isna(union):
                single_sensors = temp_rows.loc[(temp_rows["Unions"].isnull()), ["External Tags"]]
                single_sensors = single_sensors["External Tags"].to_list()
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
                        union_rows = temp_rows.loc[(temp_rows["Unions"].str.contains(temp, regex=False) == True)]
                        print(union_rows)
                        union_dict = {
                            str(int(temp[:temp.find('(')])): {
                                "name": union_rows["Name"].iloc[0],
                                "sensors": "null",
                            }
                        }
                        print(union_dict)
                        union_sensors = union_rows["External Tags"].to_list()
                        if group_dict[str(int(group))]["unions"] == "null":
                            group_dict[str(int(group))]["unions"] = []
                        union_dict[str(int(temp[:temp.find('(')]))]["sensors"] = union_sensors
                        group_dict[str(int(group))]["unions"].append(union_dict)
                        marked_union.append(temp)
        json_dict["groups"].append(group_dict)

    # Непомеченные группы
    no_group_sensors = data_excel.loc[(pd.isna(data_excel["Groups"]))]["External Tags"].to_list()
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
def unite(dict, df_excel_path):
    df = pd.read_excel(df_excel_path)
    #df.rename(columns={"External Tags": "timestamp"}, inplace=True)
    time = df["timestamp"]
    data = df.drop(columns=["timestamp"])

    print(data)
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