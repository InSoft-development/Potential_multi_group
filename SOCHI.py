import json
import os
import argparse

import pandas as pd
import sqlite3
import clickhouse_connect

import union
import normalization


DATA_DIR = f'Data'


def create_parser():
    parser = argparse.ArgumentParser(description="prepare and normalize data")
    parser.add_argument("-v", "--version", action="version", help="print version", version="1.0.1")
    return parser


# Функция автоматической подготовки данных для обучения с отсройкой всех параметров
def prepare_train_multi_regress():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    blacklist = config_json['model']['blacklist_sensors']
    approxlist = config_json['model']['approx_sensors']
    source_data = config_json["source_input_data"]

    # Чтение ненормализованного и необъединенного файла csv
    if source_data == "clickhouse":
        print("source from clickhouse")
        client = clickhouse_connect.get_client(host=config_json['paths']['database']['clickhouse']['host_ip'],
                                               username=config_json['paths']['database']['clickhouse']['username'],
                                               password=config_json['paths']['database']['clickhouse']['password'])
        df_original = client.query_df(config_json['paths']['database']['clickhouse']['original_csv_query'])
        df_original.drop(columns=blacklist, inplace=True)
        client.close()
    elif source_data == "sqlite":
        print("source from sqlite")
        client = sqlite3.connect(config_json['paths']['database']['sqlite']['original_csv'])
        df_original = pd.read_sql_query(config_json['paths']['database']['sqlite']['original_csv_query'], client)
        df_original.drop(columns=blacklist, inplace=True)
        client.close()
    elif source_data == "csv":
        print("source from csv")
        path_to_original_csv = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['original_csv']}"
        df_original = pd.read_csv(path_to_original_csv)
        df_original.drop(columns=blacklist, inplace=True)
    else:
        print("complete field source_input_data in config (possible value: clickhouse, sqlite, csv) and rerun script")
        exit(0)
    # Сохранение файла json с маркированными датчиками
    path_save_json = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['json_sensors']}"

    # if config_json["source_input_data"] == "csv":
    #     path_kks = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['original_kks']}"
    #     df_original_kks = pd.read_csv(path_kks, delimiter=';', header=None)
    #     df_original_kks = df_original_kks[~(df_original_kks[0].isin(blacklist))]
    #     print(df_original_kks)

    # Объединение датчиков и их нормализация с отстройками по мощностям и отсечка по уровню мощности
    union.json_build(source_data, config_json, path_save_json)
    path_truncate_power_df = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['csv_truncate_by_power']}"
    path_train_df = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['csv_train']}"
    df_original = df_original.loc[df_original[approxlist[len(approxlist)-1]] > config_json['model']['N']]

    df_TRAIN = df_original.copy(deep=True)
    df_TRAIN = df_original.loc[1:len(df_TRAIN):100, :]
    df_TRAIN.reset_index(drop=True)
    df_original.to_csv(path_truncate_power_df, index=False)
    df_TRAIN.to_csv(path_train_df, index=False)
    print("success " + path_truncate_power_df)

    with open(path_save_json, 'r', encoding='utf8') as j:
        unions_json = json.load(j)

    # Получение коэффициентов и нормализованного dataframe c unions
    path_coef_train_json = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['coef_train_json']}"

    if len(approxlist) == 1:
        normalization.normalize_multi_regress_one_powers(union.unite(unions_json, path_train_df),
                                                         path_coef_train_json, approxlist[0])
        df_norm = normalization.normalize_multi_regress_one_powers_df(union.unite(unions_json, path_truncate_power_df),
                                                                      path_coef_train_json, approxlist[0])
    else:
        normalization.normalize_multi_regress_two_powers(union.unite(unions_json, path_train_df),
                                                         path_coef_train_json, approxlist[1],
                                                         approxlist[0])
        df_norm = normalization.normalize_multi_regress_two_powers_df(union.unite(unions_json, path_truncate_power_df),
                                                                      path_coef_train_json, approxlist[1],
                                                                      approxlist[0])
    df_norm.drop(columns=approxlist, inplace=True)
    print("************************")
    print(df_norm)
    path_save_sqlite = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['sqlite_norm']}"
    connection = sqlite3.connect(path_save_sqlite)
    df_norm.to_sql("data", connection, if_exists='replace', index=False)
    connection.close()
    print("success " + path_save_sqlite)


if __name__ == '__main__':
    parser = create_parser()
    namespace = parser.parse_args()

    with open("config_SOCHI.json", 'r', encoding='utf8') as j:
        config_json = json.load(j)

    prepare_train_multi_regress()
