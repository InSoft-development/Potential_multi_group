# Potential_multi_group

## Назначение

Набор скриптов реализует метод потенциалов, предназначенный для обнаружения и прогнозирования аномалий в ДТО (диагностируемом техническом оборудовании). Каждый скрипт реализует определенный этап метода потенциалов и запускается последовательно. Для запуска всех скриптов необходим конфигурационный файл json, csv файлы с временными срезами по датчикам и принадлежностью датчика к определенной группе.

### В состав набора входят следующие скрипты:

- SOCHI.py: отбор по мощности срезов для обучения, объединение и нормализация;
- union.py: функции объединения датчикво по группам;
- normalization.py: функции нормализации фреймов данных;
- potential_train_json.py: отбор датчиков и точек для вычисления потенциала;
- potential_analyze_period_json.py: вычисление потенцила и loss;
- P_limit_improvement.py: перевод потенциала в вероятностное пространство;
- anomaly_time_predict_prob.py: вычисление времени до аномалии через вероятнсоть;
- anomaly_time_predict_intercept.py: вычисление времени до аномалии через свободный член регрессии;
- potential_intervals_json.py: выделение аномальных интервалов в формате json;
- potentials_online_clickhouse: онлайн режим метода потенциалов.


### Системные требования 

- pandas~=1.4.3;
- loguru~=0.5.3;
- numpy~=1.23.1;
- matplotlib~=3.5.2;
- scikit-learn~=1.1.1;
- scipy~=1.9.0;
- python-dateutil~=2.8.2;
- clickhouse_connect~=0.5.18.


### Установка пакетов
```
pip install -r requirements.txt
```
## Запуск скриптов

Для офлайн режима скрипты следует выполнять в следующей последовательности:

1) SOCHI.py;
2) potential_train_json.py;
3) potential_analyze_period_json.py;
4) P_limit_improvement.py;
5) anomaly_time_predict_prob.py или anomaly_time_predict_intercept.py;
6) potential_intervals_json.py

Запуск скриптов можно выполнить без аргументов, так как все необходимые параметры считываются через конфиг:
```
python SOCHI.py
python potential_train_json.py
python potential_analyze_period_json.py
python P_limit_improvement.py
python anomaly_time_predict_prob.py ИЛИ anomaly_time_predict_intercept.py;
python potential_intervals_json.py
```

Для онлайн режима необходимо выполнить:
```
python potentials_online_clickhouse
```

### Результаты выполнения скритов

В результате последовательного выполнения скриптов в директории Data будут появляться директории, содержащие json и csv файлы с результатами работы метода.
В директории json_interval будут сохранены найденные аномальные интервалы; в директориях соответствующих групп появятся csv с вычисленными потенциалами, вероятностями.

### Описание конфигурационного файла config_SOCHI.json:

#### `Предупреждение` - наименование всех сохраняемых и входных файлов лучше не изменять.

Конфигурационный файл json имеет по умолчанию следующие поля:

`station`: наименование станции.

`project_name`: название проекта.

`count_of_groups`: количество групп ДТО.

`number_of_samples`: количестов интервалов кратных 5.

`create_online_table`: если 1, то пересоздает и заполняет заново таблицы в БД для онлайн режима; если 0, то записи кладутся в существующую БД.

`source_input_data`: выбор источника входных данных. Следует указать какое-либо одно из следующих трех значений: {"clickhouse", "sqlite", "csv"}.

`N`: отсечка по мощности.

`blacklist_sensors`: массив KKS запрещенных датчиков - эти датчики не учавствуют в работе метода.

`approx_sensors`: массив KKS отсроечных датчиков. Может содержать два датчика, тогда отсройка будет от двух параметров. Если внесен в массив один датчик, то отсройка идет от одного датчика. В массиве последним `всегда` указывается KKS датчика `мощности`.

`N_l`: количество точек для вычисления потенциала.

`delta`: время назад регрессии при вычислении времени до аномалии.

`s`: интервал взятия среза при вычислении времени до аномалии.

`P_pr`: уставка сигнализации по прогнозиреумой вероятности.

`T_pr`: уставка сигнализации по расчтному времени до аномалии.

`delta_tau_P`: выдержка в часах для уставки сигнализации по вероятности.

`delta_tau_T`: выдержка в часах для уставки сигнализации по прогнозируемому времени.

`rolling`: сглаживание в часах.

`original_csv`: наименование csv файла с временными срезами значений датчиков всех групп (используется если `source_input_data`: "csv").

`original_kks`: наименование csv файла с KKS и группами (используется если `source_input_data`: "csv").

`original_group_csv`: наименование csv файла с KKS и группами (используется если `source_input_data`: "csv").

`json_sensors`: наименование json файла с выдленными скриптом группами ДТО и датчиками.

`csv_truncate_by_power`: наименование csv файла с срезами, прошедшими отсчеку по мощности.

`csv_train`: наименование csv файла с срезами для обучения коэффициентов отстройки по мощности.

`coef_train_json`: наименование json файла с коэффициентами регрессии.

`sqlite_norm`: наименование sqlite файла с нормализованными данными.

`points_json`: наименование json файла с отобранными точками.

`potentials_csv`: наименование csv файла с вычисленными потенциалами.

`loss_csv`: наименование csv файла с вычисленными loss.

`probability_csv`: наименование csv файла с рассчитанными вероятностями.

`table_potential_probability`: наименование csv файла-таблцы распределения потенцила вероятности.

`threshold_json`: наименование json файла с пороговыми значениями.

`anomaly_time_prob`: наименование csv файла с вычисленными временем до аномалии через вероятность.

`save_models_prob`: наименование json файла с коэффициентами регрессии при вычислении времени до аномалии через вероятность.

`anomaly_time_intercept`: наименование csv файла с вычисленными временем до аномалии через свободный член регрессии.

`save_models_intercept`: наименование json файла с коэффициентами регрессии при вычислении времени до аномалии через свободный член регрессии.

`intervals_json`: наименование json файла с выделенными аномальными интервалами.

`clickhouse`: объект содержит запросы к БД clichouse в случае выбора этой БД в качестве источника входных данных в поле `source_input_data`:

- `original_csv_query`: запрос на получение временных срезов из clichouse;

- `original_kks_query`: запрос на получение KKS датчиков и принадлежностей их к группам из clichouse;

- `original_group_query`: запрос на получение наименования всех групп из clichouse.

`sqlite`: объект содержит запросы к БД sqlite в случае выбора этой БД в качестве источника входных данных в поле `source_input_data`:

- `original_csv`: наименование sqlite файла, содержащего временные срезы;

- `original_csv_query`: запрос на получение временных срезов из sqlite;

- `original_kks`: наименование sqlite файла, содержащего KKS датчиков и группы;

- `original_kks_query`: запрос на получение KKS датчиков и принадлежностей их к группам из sqlite;

- `original_group`: наименование sqlite файла, содержащего наименования всех групп;

- `original_group_query`: запрос на получение наименования всех групп из sqlite.

`hist_potential_probability`: наименование png файла с гистограммой распределения потенцила и вероятсности.
