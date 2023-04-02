import json

import pandas as pd
import union
import normalization


def online_SOCHI():
    with open(config_json['paths']['files']['original_csv'], 'r') as f:
        # Чтение ненормализованного и необъединенного файла csv
        df_original = pd.read_csv(f.name)  # тест для построчного перебора
        blacklist = config_json['model']['blacklist_sensors']
        approxlist = config_json['model']['approx_sensors']
        for index, row in df_original.iterrows():
            row.drop(columns=blacklist, inplace=True)
            # Отсечка по мощности
            if row[approxlist[1]] <= config_json['model']['N']:
                continue

            # Объединение, отстройка и нормализация
            row_union = union.unite_row('sensors.json', row.to_dict())
            row_norm = normalization.normalize_multi_regress_two_powers_row(row_union, 'slices_GT_coef_train.json',
                                                                            approxlist[1], approxlist[0])
            # Выбрасываем отстроечные параметры из словаря
            del row_norm[approxlist[0]]
            del row_norm[approxlist[1]]


if __name__ == '__main__':
    with open("config_SOCHI.json", 'r', encoding='utf8') as j:
        config_json = json.load(j)
    online_SOCHI()
    #prepare_train_multi_regress()


# import time
# import clickhouse_connect
# import pandas as pd
#
#
# client = clickhouse_connect.get_client(host='10.23.0.177', username='default', password='asdf')
# client.command("DROP TABLE IF EXISTS sum_row")
# client.command("CREATE TABLE sum_row (sum Float64, timestamp DateTime64) ENGINE = " \
#       "MergeTree() PARTITION BY toYYYYMM(timestamp) ORDER BY (timestamp) PRIMARY KEY (timestamp)")
# try:
#     while True:
#         last_row = client.query_df("SELECT * from slices_play order by timestamp desc limit 1")
#         print(last_row)
#         s = last_row.sum(axis=1, numeric_only=True).iloc[0]
#         t = last_row["timestamp"].iloc[0]
#         client.insert_df("sum_row", pd.DataFrame([{"sum": s, "timestamp": t}]))
#         new_records = client.query_df("SELECT * FROM sum_row").tail(1)
#         print(new_records)
#         time.sleep(5)
# finally:
#     print("disconnected")
