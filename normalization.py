import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import json


# создание соединения с sqlite
def connect_sqlite(db_file):
    try:
        connection = sqlite3.connect(db_file)
        return connection
    except sqlite3.Error as error:
        print("Error has been occurred in connection to sqlite", error)


# Нормализация и отстройка всего dataframe, содержащего unions, сохранение коэффициентов в json
def normalize(df, power, path_to_save):
    time = df["timestamp"]
    data = df.drop(columns=["timestamp"])
    data.dropna(axis=1, how='all', inplace=True)
    avg = data.mean()
    data.fillna(avg, inplace=True)
    sigma = data.std()
    data -= avg
    data = data / (3 * sigma)
    print("normalized\n", data)

    # Отстройка по мощности
    models_json = {}
    data_corr = pd.DataFrame()

    for d in data:
        regr = LinearRegression().fit(X=data[power].values.reshape(-1, 1), y=data[d].values.reshape(-1, 1))
        models_json[str(d)] = ({"k": (1 - float(regr.coef_)) / (3 * sigma[d]),
                                "c": (avg[d] * (float(regr.coef_) - 1)) / (3 * sigma[d]) - float(regr.intercept_)})
        print(float(regr.coef_))
        data_corr[d] = regr.predict(data[power].values.reshape(-1, 1)).reshape(-1)
    data -= data_corr

    # сохранение коэффициентов в json
    with open(path_to_save, 'w', encoding='utf8') as f:
        json.dump(models_json, f, ensure_ascii=False, indent=4)

    data.insert(0, "timestamp", time)
    print(data)
    return data


# Отстройка и нормализация dataframe, содержащего unions, от двух параметров по известным коэффициентам - по Коршиковой
def normalize_two_powers_df(df_union, station_path_1, station_path_2, approx_1, approx_2):
    df_norm = df_union.copy(deep=True)
    try:
        len(normalize_two_powers_df.coef_1)
        len(normalize_two_powers_df.coef_2)
    except AttributeError:
        print("read json")
        with open(station_path_1, 'r', encoding='utf8') as f:
            coef_json_1 = json.load(f)
        normalize_two_powers_df.coef_1 = coef_json_1
        with open(station_path_2, 'r', encoding='utf8') as f:
            coef_json_2 = json.load(f)
        normalize_two_powers_df.coef_2 = coef_json_2
    df_norm_1 = pd.DataFrame()
    df_norm_2 = pd.DataFrame()
    for key in list(normalize_two_powers_df.coef_1.keys()):
        df_norm_1[key] = df_union[approx_1] * normalize_two_powers_df.coef_1[key]['k'] + \
                         normalize_two_powers_df.coef_1[key]['c']
        df_norm[key] -= df_norm_1[key]
    for key in list(normalize_two_powers_df.coef_2.keys()):
        df_norm_2[key] = df_union[approx_2] * normalize_two_powers_df.coef_2[key]['k'] + \
                         normalize_two_powers_df.coef_2[key]['c']
        df_norm[key] -= df_norm_2[key]
    time = df_norm["timestamp"]
    df_norm = df_norm.drop(columns=["timestamp"])
    df_norm.dropna(axis=1, how='all', inplace=True)
    avg = df_norm.mean()
    df_norm.fillna(avg, inplace=True)
    sigma = df_norm.std()
    df_norm -= avg
    df_norm = df_norm / (3 * sigma)
    df_norm.insert(0, "timestamp", time)
    return df_norm


# Отстройка dataframe, содержащего unions, от всех параметров, по коэффициентам из файла json
def normalize_df(df_union, station_path):
    df_norm = df_union
    try:
        len(normalize_df.coef)
    except AttributeError:
        print("read json")
        with open(station_path, 'r', encoding='utf8') as f:
            coef_json = json.load(f)
        normalize_df.coef = coef_json
    for key in list(normalize_df.coef.keys()):
        df_norm[key] = df_union[key] * normalize_df.coef[key]['k'] + normalize_df.coef[key]['c']
    return df_norm


# Функция выполняет отстройку union объединения для одной строки данных
def normalize_row(station_path, dict_union):
    dict_norm = {}
    try:
        len(normalize_row.coef)
    except AttributeError:
        print("read json")
        with open(station_path, 'r', encoding='utf8') as f:
            coef_json = json.load(f)
        normalize_row.coef = coef_json
    for key in list(normalize_row.coef.keys()):
        dict_norm[key] = dict_union[key] * normalize_row.coef[key]['k'] + normalize_row.coef[key]['c']
    dict_norm['timestamp'] = dict_union['timestamp']
    return dict_norm


# Функция выполняет отстройку и нормализацию union объединения с сохранением коэффициентов регрессии для одного
# внешнего параметра
def normalize_multi_regress_one_powers(df_union, path_to_save, approx):
    # Подготовка к отстройке
    df_norm = df_union.copy(deep=True)
    time = df_norm["timestamp"]
    df_norm = df_norm.drop(columns=["timestamp"])
    df_norm.dropna(axis=1, how='all', inplace=True)

    # Отстройка по мощности
    models_json = {}

    for d in df_norm:
        Y = df_union[d].to_numpy()
        X = df_union[approx].to_numpy()
        X = np.c_[X, np.ones(X.shape[0])]
        beta_hat = np.linalg.lstsq(X, Y, rcond=None)
        # Вычитаем отстройку
        df_norm[d] -= np.dot(X, beta_hat[0])
        # Нормализуем столбец
        avg = df_norm[d].mean()
        df_norm[d].fillna(avg, inplace=True)
        sigma = df_norm[d].std()
        df_norm[d] -= avg
        df_norm[d] = df_norm[d] / (3 * sigma)
        models_json[str(d)] = ({"a": beta_hat[0][0], "c": beta_hat[0][1],
                                "avg": avg, "sigma": sigma})
    # сохранение коэффициентов в json
    with open(path_to_save, 'w', encoding='utf8') as f:
        json.dump(models_json, f, ensure_ascii=False, indent=4)
    df_norm.insert(0, "timestamp", time)
    return df_norm


# Функция выполняет отстройку и нормализацию всего датафрейма union объединения по
# сохраненным коэффициентам регрессии для одного внешнего параметра
def normalize_multi_regress_one_powers_df(df_union, path_to_json, approx):
    df_norm = df_union.copy(deep=True)
    try:
        len(normalize_multi_regress_one_powers_df.coef_df)
    except AttributeError:
        print("read json")
        with open(path_to_json, 'r', encoding='utf8') as f:
            coef_json = json.load(f)
        normalize_multi_regress_one_powers_df.coef_df = coef_json
    for key in list(normalize_multi_regress_one_powers_df.coef_df.keys()):
        df_norm[key] -= df_union[approx] * normalize_multi_regress_one_powers_df.coef_df[key]['a'] + \
                        normalize_multi_regress_one_powers_df.coef_df[key]['c']
        df_norm[key].fillna(normalize_multi_regress_one_powers_df.coef_df[key]['avg'], inplace=True)
        df_norm[key] -= normalize_multi_regress_one_powers_df.coef_df[key]['avg']
        df_norm[key] = df_norm[key] / (3 * normalize_multi_regress_one_powers_df.coef_df[key]['sigma'])
    return df_norm


# Функция выполняет отстройку и нормализацию одной строки union объединения по
# сохраненным коэффициентам регрессии для одного внешнего параметра
def normalize_multi_regress_one_powers_row(dict_union, path_to_json, approx):
    dict_norm = {}
    try:
        len(normalize_multi_regress_one_powers_row.coef_row)
    except AttributeError:
        print("read json")
        with open(path_to_json, 'r', encoding='utf8') as f:
            coef_json_row = json.load(f)
        normalize_multi_regress_one_powers_row.coef_row = coef_json_row
    dict_norm['timestamp'] = dict_union['timestamp']
    for key in list(normalize_multi_regress_one_powers_row.coef_row.keys()):
        print(dict_union[key])
        dict_norm[key] = dict_union[key]-\
                         dict_union[approx]*normalize_multi_regress_one_powers_row.coef_row[key]['a']-\
                         normalize_multi_regress_one_powers_row.coef_row[key]['c']
        print(dict_norm[key])
        dict_norm[key] -= normalize_multi_regress_one_powers_row.coef_row[key]['avg']
        print(dict_norm[key])
        dict_norm[key] = dict_norm[key] / (3 * normalize_multi_regress_one_powers_row.coef_row[key]['sigma'])
        print(dict_norm[key])
    return dict_norm


# Функция выполняет отстройку и нормализацию union объединения с сохранением коэффициентов мультирегрессии
def normalize_multi_regress_two_powers(df_union, path_to_save, approx_1, approx_2):
    # Подготовка к отстройке
    df_union.dropna(axis=1, how='all', inplace=True)
    df_union.fillna(df_union.mean(), inplace=True)
    df_norm = df_union.copy(deep=True)
    time = df_norm["timestamp"]
    df_norm = df_norm.drop(columns=["timestamp"])
    df_norm.dropna(axis=1, how='all', inplace=True)
    df_norm.fillna(df_norm.mean(), inplace=True)

    # Отстройка по мощности
    models_json = {}

    for d in df_norm:
        Y = df_union[d].to_numpy()
        X = df_union[[approx_1, approx_2]].to_numpy()
        X = np.c_[X, np.ones(X.shape[0])]
        print(X)
        print(Y)
        beta_hat = np.linalg.lstsq(X, Y, rcond=None)
        # Вычитаем отстройку
        df_norm[d] -= np.dot(X, beta_hat[0])
        # Нормализуем столбец
        avg = df_norm[d].mean()
        df_norm[d].fillna(avg, inplace=True)
        sigma = df_norm[d].std()
        df_norm[d] -= avg
        df_norm[d] = df_norm[d] / (3 * sigma)
        models_json[str(d)] = ({"a": beta_hat[0][0], "b": beta_hat[0][1], "c": beta_hat[0][2],
                                "avg": avg, "sigma": sigma})
    # сохранение коэффициентов в json
    with open(path_to_save, 'w', encoding='utf8') as f:
        json.dump(models_json, f, ensure_ascii=False, indent=4)
    df_norm.insert(0, "timestamp", time)
    return df_norm


# Функция выполняет отстройку и нормализацию всего датафрейма union объединения по
# сохраненным коэффициентам мультирегрессии
def normalize_multi_regress_two_powers_df(df_union, path_to_json, approx_1, approx_2):
    df_norm = df_union.copy(deep=True)
    try:
        len(normalize_multi_regress_two_powers_df.coef_df)
    except AttributeError:
        print("read json")
        with open(path_to_json, 'r', encoding='utf8') as f:
            coef_json = json.load(f)
        normalize_multi_regress_two_powers_df.coef_df = coef_json
    for key in list(normalize_multi_regress_two_powers_df.coef_df.keys()):
        df_norm[key] -= df_union[approx_1] * normalize_multi_regress_two_powers_df.coef_df[key]['a'] + \
                        df_union[approx_2] * normalize_multi_regress_two_powers_df.coef_df[key]['b'] + \
                        normalize_multi_regress_two_powers_df.coef_df[key]['c']
        df_norm[key].fillna(normalize_multi_regress_two_powers_df.coef_df[key]['avg'], inplace=True)
        df_norm[key] -= normalize_multi_regress_two_powers_df.coef_df[key]['avg']
        df_norm[key] = df_norm[key] / (3 * normalize_multi_regress_two_powers_df.coef_df[key]['sigma'])
    return df_norm


# Функция выполняет отстройку и нормализацию одной строки union объединения по
# сохраненным коэффициентам мультирегрессии
def normalize_multi_regress_two_powers_row(dict_union, path_to_json, approx_1, approx_2):
    dict_norm = {}
    try:
        len(normalize_multi_regress_two_powers_row.coef_row)
    except AttributeError:
        print("read json")
        with open(path_to_json, 'r', encoding='utf8') as f:
            coef_json_row = json.load(f)
        normalize_multi_regress_two_powers_row.coef_row = coef_json_row
    dict_norm['timestamp'] = dict_union['timestamp']
    for key in list(normalize_multi_regress_two_powers_row.coef_row.keys()):
        print(dict_union[key])
        dict_norm[key] = dict_union[key] - dict_union[approx_1]*normalize_multi_regress_two_powers_row.coef_row[key]['a'] -\
                         dict_union[approx_2]*normalize_multi_regress_two_powers_row.coef_row[key]['b'] - \
                         normalize_multi_regress_two_powers_row.coef_row[key]['c']
        print(dict_norm[key])
        dict_norm[key] -= normalize_multi_regress_two_powers_row.coef_row[key]['avg']
        print(dict_norm[key])
        dict_norm[key] = dict_norm[key] / (3 * normalize_multi_regress_two_powers_row.coef_row[key]['sigma'])
        print(dict_norm[key])
    return dict_norm
