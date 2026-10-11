import pandas as pd
import numpy as np
import os

# Гарантируем, что папка data существует
os.makedirs('data', exist_ok=True)

np.random.seed(42)
n = 300

df = pd.DataFrame({
    'startup_cost': np.random.uniform(0, 50, n),
    'total_cost': np.random.uniform(100, 5000, n),
    'plan_rows': np.random.randint(10, 20000, n),
    'plan_width': np.random.randint(16, 128, n),
    'join_nodes_cnt': np.random.randint(1, 6, n),
    'scan_nodes_cnt': np.random.randint(1, 8, n),
    'index_scans_cnt': np.random.randint(0, 5, n),
    'filter_selectivity': np.random.uniform(0.01, 1.0, n)
})

df['actual_time_ms'] = (df['total_cost'] * 0.04 + df['join_nodes_cnt'] * 12 + np.random.normal(0, 15, n)).clip(lower=0.1)
df.to_csv('data/dataset.csv', index=False)
print('Датасет планов v1.0 успешно создан в папке data/')
