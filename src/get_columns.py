import pandas as pd
def get_columns():
    df = pd.read_csv('../data/homework.csv').drop('price_category', axis=1)
    types = {
        'int64': 'int',
        'float64': 'float'
    }

    for k, v in df.dtypes.iteritems():
        print(f'{k}: {types.get(str(v), "str")}')