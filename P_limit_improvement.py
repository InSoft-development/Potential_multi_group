import argparse
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy.stats
import os

DATA_DIR = f'Data'


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_df", nargs=1, help="path to CSV dataframe with potentials")
    parser.add_argument("path_to_power", nargs=1, help="path to xlsx with unnormalized power")
    parser.add_argument("path_to_probability", nargs=1, help="path to saving CSV with probabilities, potentials")
    parser.add_argument("path_to_csv", nargs=1, help="path to saving CSV with table potential & probability")
    parser.add_argument("path_to_png", nargs=1, help="path to saving png picture with hist potential & probability")
    parser.add_argument("path_to_threshold", nargs=1, help="path to saving json with thresholds")
    parser.add_argument("-power", nargs='+', help="sensor of power: N should be typed in the end", required=True)
    return parser


def p_limit_one_power(power_sensor):
    df_potential = pd.read_csv(path_to_df)
    df_power = pd.read_csv(path_to_power)
    # массивы значений датчика мощности
    rotor = df_power[power_sensor].tolist()
    print(df_potential)

    t = df_potential['timestamp'].tolist()  # Массив временных отсчетов
    a = df_potential['potential'].tolist()  # Массив значений потенциалов

    data = pd.DataFrame({"timestamp": t, "potential": a, "N": rotor})

    #data_train = data.loc[1:78260:100, :]
    data_train = data.loc[1:len(data):100, :]
    print(data_train)

    hist = np.histogram(data_train['potential'].values, bins=100)
    dist = scipy.stats.rv_histogram(hist)

    d = np.arange(min(data_train['potential']), max(data_train['potential']), 0.001)
    potentials = 100 * dist.pdf(d)
    probabilities = 100 * (1 - dist.cdf(d))
    plt.plot(d, probabilities, 'g')
    plt.plot(d, potentials, 'r')
    data['P'] = [100 * (1 - i) for i in dist.cdf(data['potential'].values)]
    data.to_csv(path_to_probability)
    print("limit ok")

    df = pd.DataFrame(data={'potential': d, 'probability': probabilities}, index=None)
    df.to_csv(path_to_csv, index=False)
    temp = df.index[(df['probability'] < config_json['model']['P_pr'] * 100 + 1)].tolist()
    print(temp)
    ind = temp[0]
    print("threshold = ", d[ind])
    threshold_json[group] = d[ind]
    plt.axvline(d[ind], color='brown')
    plt.title("probability and hist of potential")
    plt.legend(["probability", "hist of potential", "threshold"])
    plt.savefig(path_to_png)
    plt.show()
    plt.clf()
    print(df)


def p_limit_two_power(power_sensor_1, power_sensor_2):
    df_potential = pd.read_csv(path_to_df)
    df_power = pd.read_csv(path_to_power)
    # массивы значений датчика мощности
    rotor_1 = df_power[power_sensor_1].tolist()
    rotor_2 = df_power[power_sensor_2].tolist()
    print(df_potential)

    t = df_potential['timestamp'].tolist()  # Массив временных отсчетов
    a = df_potential['potential'].tolist()  # Массив значений потенциалов

    data = pd.DataFrame({"timestamp": t, "potential": a, "T": rotor_1, "N": rotor_2})

    #data_train = data.loc[1:78260:100, :]
    data_train = data.loc[1:len(data):100, :]
    print(data_train)

    hist = np.histogram(data_train['potential'].values, bins=100)
    dist = scipy.stats.rv_histogram(hist)

    d = np.arange(min(data_train['potential']), max(data_train['potential']), 0.001)
    potentials = 100 * dist.pdf(d)
    probabilities = 100 * (1 - dist.cdf(d))
    plt.plot(d, probabilities, 'g')
    plt.plot(d, potentials, 'r')
    data['P'] = [100 * (1 - i) for i in dist.cdf(data['potential'].values)]
    data.to_csv(path_to_probability)
    print("limit ok")

    df = pd.DataFrame(data={'potential': d, 'probability': probabilities}, index=None)
    df.to_csv(path_to_csv, index=False)
    temp = df.index[(df['probability'] < config_json['model']['P_pr'] * 100 + 1)].tolist()
    print(temp)
    ind = temp[0]
    print("threshold = ", d[ind])
    threshold_json[group] = d[ind]
    plt.axvline(d[ind], color='brown')
    plt.title("probability and hist of potential")
    plt.legend(["probability", "hist of potential", "threshold"])
    plt.savefig(path_to_png)
    plt.clf()
    #plt.show()
    print(df)


if __name__ == '__main__':
    parser = create_parser()
    with open("config_SOCHI.json", 'r', encoding='utf8') as j:
        config_json = json.load(j)
    if len(sys.argv) == 1:
        print("config SOCHI")
        #path_to_df = "1\\" + config_json['paths']['files']['potentials_csv']
        path_to_power = f"{DATA_DIR}{os.sep}{config_json['paths']['files']['csv_truncate_by_power']}"
        #path_to_probability = config_json['paths']['files']['probability_csv']
        #path_to_csv = config_json['paths']['files']['table_potential_probability']
        #path_to_png = config_json['paths']['graphs']['hist_potential_probability']
        #path_to_threshold_json = config_json['paths']['files']['threshold_json']
        power = config_json['model']['approx_sensors']
        with open(f"{DATA_DIR}{os.sep}{config_json['paths']['files']['json_sensors']}", 'r', encoding='utf8') as f:
            json_dict = json.load(f)

        index_group = [list(x.keys())[0] for x in json_dict["groups"]]
        if index_group[0] == '0':
            index_group.remove('0')
        if len(power) == 1:
            for group in index_group:
                path_to_probability = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                                      f"{config_json['paths']['files']['probability_csv']}{group}.csv"
                path_to_csv = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                              f"{config_json['paths']['files']['table_potential_probability']}{group}.csv"
                path_to_png = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                              f"config_json['paths']['graphs']['hist_potential_probability']{group}.png"
                path_to_threshold_json = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                                         f"{config_json['paths']['files']['threshold_json']}{group}.json"
                threshold_json = {}
                path_to_df = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                             f"{config_json['paths']['files']['potentials_csv']}{group}.csv"
                p_limit_one_power(power[0])
                with open(path_to_threshold_json, 'w', encoding='utf8') as f:
                    json.dump(threshold_json, f, ensure_ascii=False, indent=4)
        else:
            for group in index_group:
                path_to_probability = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                                      f"{config_json['paths']['files']['probability_csv']}{group}.csv"
                path_to_csv = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                              f"{config_json['paths']['files']['table_potential_probability']}{group}.csv"
                path_to_png = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                              f"{config_json['paths']['graphs']['hist_potential_probability']}{group}.png"
                path_to_threshold_json = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                                         f"{config_json['paths']['files']['threshold_json']}{group}.json"
                threshold_json = {}
                path_to_df = f"{DATA_DIR}{os.sep}{group}{os.sep}" \
                             f"{config_json['paths']['files']['potentials_csv']}{group}.csv"
                p_limit_two_power(power[0], power[1])
                with open(path_to_threshold_json, 'w', encoding='utf8') as f:
                    json.dump(threshold_json, f, ensure_ascii=False, indent=4)
    else:
        print("command's line arguments")
        namespace = parser.parse_args()
        path_to_df = namespace.path_to_df[0]
        path_to_power = namespace.path_to_power[0]
        path_to_probability = namespace.path_to_probability[0]
        path_to_csv = namespace.path_to_csv[0]
        path_to_png = namespace.path_to_png[0]
        path_to_threshold_json = namespace.path_to_threshold[0]
        power = namespace.power
        threshold_json = {}
        if len(power) == 1:
            p_limit_one_power(power[0])
        else:
            p_limit_two_power(power[0], power[1])
        with open(path_to_threshold_json, 'w', encoding='utf8') as f:
            json.dump(threshold_json, f, ensure_ascii=False, indent=4)

