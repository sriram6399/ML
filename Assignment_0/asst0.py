'''import pandas as pd
file_path = """C:/Users/gutti/Downloads/census+income+kdd (1)/census.tar/census/census-income.test"""

census_data = pd.read_csv(file_path,encoding="utf-8", delimiter=",")

#print(census_data.iloc[:,10])
age_find = census_data[census_data.iloc[:,41]== ' 50000+.']
print(age_find.iloc[:,0].mean())'''

import numpy as np
import matplotlib.pyplot as mplot
covariance_matrix = np.array([[1, -0.5], [-0.5, 1]])
mean = [0, 0]
sampled_data = np.random.multivariate_normal(mean, covariance_matrix, 100)
mplot.scatter(sampled_data[:, 0], sampled_data[:, 1])
mplot.xlabel("Horizontal axis X1")
mplot.ylabel("Vertical axis X2")
mplot.show()