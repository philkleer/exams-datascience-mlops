import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# loading data
df = pd.read_csv('data/raw/raw.csv', index_col=0)

# renaming since later in service.py we need callable columnnames
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

# splitting data
target = df['chance_of_admit']

# dropping target, SOP & LOR (because I think they are quite subjective and probably covered by the other features)
features = df.drop(['chance_of_admit', 'sop', 'lor'], axis=1)

# splitting
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# saving datasets
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print('All splits of the dataset successfully saved in \'data/processed/\' ')