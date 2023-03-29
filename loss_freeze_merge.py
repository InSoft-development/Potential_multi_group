import argparse
import os
import sys
import json
import pandas as pd


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_anomaly_time", nargs=1, help="path to CSV file with anomaly date")
    parser.add_argument("path_to_loss_csv", nargs=1, help="path to json with thresholds")
    return parser


def loss_merge_for_report(path_to_anomaly_time, path_to_loss):
    df_anomaly = pd.read_csv(path_to_anomaly_time)
    df_loss = pd.read_csv(path_to_loss)
    df_loss = pd.merge(df_anomaly['t'], df_loss, how='left', left_on='t', right_on='timestamp')
    df_loss.drop(columns=['timestamp'], inplace=True)
    df_loss.rename(columns={'t': 'timestamp'}, inplace=True)
    for index, row in df_loss.iterrows():
        if row.isnull().values.any():
            row[1:] = df_loss.iloc[index - 1][1:]
            df_loss.iloc[index] = row
    df_loss.to_csv(path_to_loss, index=False)


def loss_merge(group):
    print("group", group)
    df_anomaly = pd.read_csv(path_to_anomaly_time)
    df_loss = pd.read_csv(group + "/" + config_json['paths']['files']['loss_csv'])
    df_loss = pd.merge(df_anomaly['t'], df_loss, how='left', left_on='t', right_on='timestamp')
    df_loss.drop(columns=['timestamp'], inplace=True)
    df_loss.rename(columns={'t': 'timestamp'}, inplace=True)
    print(df_loss)
    for index, row in df_loss.iterrows():
        if row.isnull().values.any():
            row[1:] = df_loss.iloc[index - 1][1:]
            df_loss.iloc[index] = row
    print(df_loss)
    df_loss.to_csv(group + "/" + config_json['paths']['files']['loss_csv'], index=False)


if __name__ == '__main__':
    # Заполнение коэффициентов json из всего dataframe
    parser = create_parser()
    with open("config_SOCHI_generator.json", 'r', encoding='utf8') as j:
        config_json = json.load(j)
    if len(sys.argv) == 1:
        print("config SOCHI_generator")
        print("Enter 0 for choose anomaly_time_prob frame\nEnter 1 for choose anomaly_time_intercept frame")
        input_anomaly_file = input()
        if int(input_anomaly_file) == 0:
            path_to_anomaly_time = config_json['paths']['files']['anomaly_time_prob']
        if int(input_anomaly_file) == 1:
            path_to_anomaly_time = config_json['paths']['files']['anomaly_time_intercept']
        path_to_loss_csv = config_json['paths']['files']['loss_csv']
    else:
        print("command's line arguments")
        namespace = parser.parse_args()
        path_to_anomaly_time = namespace.path_to_anomaly_time[0]
        path_to_loss_csv = namespace.path_to_loss_csv[0]
    for g in range(1, config_json['count_of_groups']+1):
        loss_merge(str(g))
