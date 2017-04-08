import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic


titanic = pd.read_csv('titanic_data.csv')
titanic['AgeGroup'] = pd.cut(titanic.Age, 10, labels=False)


def survival_ratio(*args):
    frequencies = pd.crosstab(index=titanic['Survived'], columns=[titanic[arg] for arg in args])
    return frequencies.apply(func=lambda row: row / row.sum())


print(survival_ratio('Parch', 'Pclass',  'Sex'))

mosaic(titanic, ['AgeGroup', 'Survived'])
plt.show()
