import json
import os

import pandas as pd
import sqlite3

import union
import normalization


DATA_DIR = f'Data'


# Функция автоматической подготовки данных для обучения с отсройкой всех параметров
def prepare_train_multi_regress():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    # Чтение ненормализованного и необъединенного файла csv
    path_to_original_csv = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['original_csv']}"
    df_original = pd.read_csv(path_to_original_csv)
    blacklist = config_json['model']['blacklist_sensors']
    approxlist = config_json['model']['approx_sensors']
    df_original.drop(columns=blacklist, inplace=True)

    # Сохранение файла json с маркированными датчиками
    path_save_json = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['json_sensors']}"

    # path_kks = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['original_kks']}"
    # df_original_kks = pd.read_csv(path_kks, delimiter=';', header=None)
    # df_original_kks = df_original_kks[~(df_original_kks[0].isin(blacklist))]
    # print(df_original_kks)

    # Объединение датчиков и их нормализация с отстройками по мощностям и отсечка по уровню мощности
    union.json_build(path_save_json)
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
        df_TRAIN_norm = normalization.normalize_multi_regress_one_powers(union.unite(unions_json, path_train_df),
                                                                         path_coef_train_json, approxlist[0])
        df_norm = normalization.normalize_multi_regress_one_powers_df(union.unite(unions_json, path_truncate_power_df),
                                                                      path_coef_train_json, approxlist[0])
    else:
        df_TRAIN_norm = normalization.normalize_multi_regress_two_powers(union.unite(unions_json, path_train_df),
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


# Функция построения графиков параметров с отстройкой от множественной регрессии
# def multi_regress_parameters_lines():
#     path_to_df = config_json['paths']['files']['excel_df']
#     path_to_norm = config_json['paths']['files']['sqlite_norm']
#     path_to_coef_json = config_json['paths']['files']['coef_train_json']
#     approxlist = config_json['model']['approx_sensors']
#
#     df_xlsx = pd.read_excel(path_to_df)
#     print(df_xlsx)
#     with open(path_to_coef_json, 'r', encoding='utf8') as j:
#         coef_json = json.load(j)
#     print(coef_json)
#     connection = sqlite3.connect(path_to_norm)
#     df_norm = pd.read_sql_query("SELECT * from data", connection)
#     connection.close()
#     print(df_norm)
#
#     #df_regressed = df_xlsx.copy(deep=True)
#     df_regressed = pd.DataFrame()
#
#     # Графики параметров от мощности, температуры
#     for param in df_xlsx.columns[:len(df_xlsx.columns)-1]:
#         df_regressed[param] = df_xlsx[approxlist[1]] * coef_json[param]['a'] + \
#                                df_xlsx[approxlist[0]] * coef_json[param]['b'] + coef_json[param]['c']
#         plt.rcParams["figure.figsize"] = (50, 10)
#         plt.subplot(2, 1, 1)
#         plt.title(str(param))
#         plt.plot(df_xlsx[approxlist[1]], df_xlsx[param], 'o', color='blue')
#         plt.plot(df_xlsx[approxlist[1]], df_regressed[param], 'o', color='orange')
#         plt.legend(["было", "после отстройки"])
#
#         plt.subplot(2, 1, 2)
#         plt.title(str(param))
#         plt.plot(df_xlsx["timestamp"], df_xlsx[param], color='blue')
#         plt.plot(df_xlsx["timestamp"], df_regressed[param], color='orange')
#         plt.plot(df_xlsx["timestamp"], df_xlsx[param]-df_regressed[param], color='green')
#         #plt.gcf().autofmt_xdate()
#         plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=144))
#         plt.legend(["было", "отстройка", "стало"])
#         #plt.savefig('Korshikova\\Отстройка от всего\\Отстройка\\' + str(param) + '_09.11.2022.png')
#         plt.savefig(config_json['project_name'] + str(param) + '_xx.xx.2022.png')
#         plt.clf()
#         print("complete" + param)


# Функция построения графика по данным от множественной регрессии: график целиком + неделя
# def check_potential_scale_multi_regress():
#     df_zero = pd.read_csv("1\\" + config_json['paths']['files']['potentials_csv'])
#     df_probability = pd.read_csv(config_json['paths']['files']['table_potential_probability'])
#     number_of_group = [1]
#     for i in number_of_group:
#         for filename in glob.glob("1\\" + config_json['paths']['files']['potentials_csv']):
#             with open(filename, 'r') as f:
#                 print(f.name)
#                 df = pd.read_csv(f.name)
#                 df = df.loc[df['timestamp'].isin(df_zero['timestamp'])]
#                 df.reset_index(inplace=True, drop=True)
#                 probability = []
#                 for index, row in df.iterrows():
#                     find_potential = min(df_probability['potential'], key=lambda x: abs(x-row['potential']))
#                     temp_serial = df_probability['probability'].loc[df_probability['potential'] == find_potential]
#                     probability.append(temp_serial.tolist()[0])
#                 df['probability'] = probability
#                 print("probability mean: ", df['probability'].mean())
#                 print("probability median: ", df['probability'].median())
#
#                 # График потенциала целиком
#                 plt.rcParams["figure.figsize"] = (200, 10)
#                 plt.title("Potential and probability")
#                 plt.plot(df['timestamp'], df['potential'], "b-")
#                 plt.plot(df['timestamp'], df['probability']/100, "g-")
#                 plt.gcf().autofmt_xdate()
#                 plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4))
#                 plt.legend(["potentials", "probability"])
#                 plt.savefig(config_json['paths']['graphs']['scale_potential'])
#                 plt.show()
#
#                 # График мощности обучающего множества целиком
#                 plt.rcParams["figure.figsize"] = (200, 10)
#                 plt.title("Power")
#                 plt.plot(df['timestamp'], df['N'], "r-")
#                 plt.plot(df['timestamp'], df['T'], "b-")
#                 plt.gcf().autofmt_xdate()
#                 plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4))
#                 plt.legend(["N", "T"])
#                 plt.savefig(config_json['paths']['graphs']['scale_power'])
#                 plt.show()
#
#                 # График потенциала - неделя
#                 plt.rcParams["figure.figsize"] = (200, 10)
#                 plt.title("Potential and probability: week")
#                 plt.plot(df.loc[df['timestamp'] <= "2021-06-08 05:12:00"]['timestamp'],
#                          df.loc[df['timestamp'] <= "2021-06-08 05:12:00"]['potential'], "b-")
#                 plt.plot(df.loc[df['timestamp'] <= "2021-06-08 05:12:00"]['timestamp'],
#                          df.loc[df['timestamp'] <= "2021-06-08 05:12:00"]['probability'] / 100, "g-")
#                 plt.gcf().autofmt_xdate()
#                 #plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4))
#                 plt.legend(["potentials", "probability"])
#                 # plt.legend(["20CFA10CE001", "20MKA10CT015"])
#                 plt.savefig(config_json['paths']['graphs']['scale_potential_week'])
#                 plt.show()
#
#                 # График мощности обучающего множества - неделя
#                 plt.rcParams["figure.figsize"] = (200, 10)
#                 plt.title("Power: week")
#                 plt.plot(df.loc[df['timestamp'] <= "2021-06-08 05:12:00"]['timestamp'],
#                          df.loc[df['timestamp'] <= "2021-06-08 05:12:00"]['N'], "r-")
#                 plt.plot(df.loc[df['timestamp'] <= "2021-06-08 05:12:00"]['timestamp'],
#                          df.loc[df['timestamp'] <= "2021-06-08 05:12:00"]['T'], "b-")
#                 plt.gcf().autofmt_xdate()
#                 #plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4))
#                 plt.legend(["N", "T"])
#                 plt.savefig(config_json['paths']['graphs']['scale_power_week'])
#                 plt.show()


if __name__ == '__main__':
    with open("config_SOCHI.json", 'r', encoding='utf8') as j:
        config_json = json.load(j)
    prepare_train_multi_regress()
