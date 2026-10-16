import os

import numpy as np
import pandas as pd

# Проверяем, что исходный датасет существует
if not os.path.exists('data/dataset.csv'):
    raise FileNotFoundError("Исходный файл data/dataset.csv не найден! Сначала запустите генерацию v1.0.")

# Читаем первую версию
df_v1 = pd.read_csv('data/dataset.csv')

np.random.seed(100)
n = 150

# Генерируем новые планы выполнения CEB
df_ceb = pd.DataFrame({
    'startup_cost': np.random.uniform(0, 100, n),
    'total_cost': np.random.uniform(200, 12000, n),
    'plan_rows': np.random.randint(5, 50000, n),
    'plan_width': np.random.randint(16, 256, n),
    'join_nodes_cnt': np.random.randint(1, 10, n),
    'scan_nodes_cnt': np.random.randint(1, 12, n),
    'index_scans_cnt': np.random.randint(0, 8, n),
    'filter_selectivity': np.random.uniform(0.001, 0.8, n)
})

# Считаем целевую переменную с учетом новых коэффициентов
df_ceb['actual_time_ms'] = (df_ceb['total_cost'] * 0.05 + df_ceb['join_nodes_cnt'] * 18 + np.random.normal(0, 25, n)).clip(lower=0.1)

# Объединяем v1.0 и новые данные
df_v2 = pd.concat([df_v1, df_ceb], ignore_index=True)

# Сохраняем поверх
df_v2.to_csv('data/dataset.csv', index=False)
print('Датасет успешно обновился до v2.0 (JOB + CEB) и сохранен в data/dataset.csv')
