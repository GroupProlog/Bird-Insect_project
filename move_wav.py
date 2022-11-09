import shutil, os
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

meta = pd.read_csv('xcmeta.csv', sep='\t', lineterminator='\n')

sort_order = []

for file_name in os.listdir('bird class wavs'):
	file_id = file_name[2:-4]
	sort_order.append(file_id)

df_mapping = pd.DataFrame({
    'ind': sort_order,
})
sort_mapping = df_mapping.reset_index().set_index('ind')

print(sort_mapping['index'])

meta['id_num'] = list(sort_mapping['index'])

print(meta[:5])

meta = meta.sort_values('id_num')

print(meta[:5])