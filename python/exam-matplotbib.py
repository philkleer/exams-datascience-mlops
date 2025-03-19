# (a) Load the pandas package as pd and the matplotlib.pyplot sub-package as plt.
# (b) Read the house_pricing.csv file into a DataFrame called df.
# (c) Display the first 5 rows of the df DataFrame.

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('house_pricing.csv')

df.head()


# (d) What is the percentage of missing values in the dataset ?
total_missing = df.isna().sum().sum()

total_values = df.size

missing_perc = (total_missing / total_values) * 100

print(missing_perc, '% of the cells have NA')

# Histograms
# (a) Using a histogram, analyze the distribution of prices on the market.
#  We will separate the values taken by SalePrice into 30 consecutive distinct
#  intervals of the same length.
# (b) Use a pleasant and easy-to-understand style. Don't forget to add a
#  legend and a title to your graphs!
plt.hist(df['SalePrice'], bins=30, label='Sales Prices')
plt.title('Sale Prices on the market')
plt.legend()
# (c) Execute the following cell to normalize the data of `SalePrice` and
#  `GrLivArea` according to `Min Max` normalization.
df['SalePrice'] = (df['SalePrice'] - df['SalePrice'].min()) / (
    df['SalePrice'].max() - df['SalePrice'].min()
)
df['GrLivArea'] = (df['GrLivArea'] - df['GrLivArea'].min()) / (
    df['GrLivArea'].max() - df['GrLivArea'].min()
)

df['SalePrice'] = (df['SalePrice'] - df['SalePrice'].min()) / (
    df['SalePrice'].max() - df['SalePrice'].min()
)
df['GrLivArea'] = (df['GrLivArea'] - df['GrLivArea'].min()) / (
    df['GrLivArea'].max() - df['GrLivArea'].min()
)


# (d) On the same plot, display the distribution histograms of the
#  two variables SalePrice and GrLivArea. Make sure to label your plot appropriately.
plt.hist(
    [df['SalePrice'], df['GrLivArea']],
    bins=30,
    label=['Sales Prices', 'Price for Living Area'],
    color=['#f27486', '#f7c548'],
)
plt.title('Normalized Distribution (min-max)')
plt.legend()
# Alternativ
df.plot.hist(
    y=['SalePrice', 'GrLivArea'],
    bins=30,
    label=['Sales Prices', 'Living Area'],
    color=['#f27486', '#f7c548'],
    subplots=True,
)

# (e) Analyze the above figure.
# From the plot we can see that both distributions are slightly right-skewed with a top not in the center
# From the plot we can see that both distributions are slightly right-skewed with a top not in the center
# (it's rather on the left side). This means that the modus is probably higher than (or close to) the median and
# these two are higher than the mean. Therefore, interpretation of the mean might be somewhat misleading.
# Both distributions have the peak around 0.2 and show significant extreme values on the right hand-side. Therefore,
# there exist some values extremely high compared to the rest of the values of each feature. It might be worthy, to
# check if there are some errors in the distribution or if these extremely high values are indeed valid.

# It is also possible to create two-dimensional histograms with the hist2d
#  function from the matplotlib.pyplot package. The principle is as follows : the abscissa and ordinate represent the two variables to be tested and the color of the box represents the number of occurrences.

# The hist2d function takes as arguments the two variables to be studied and
#  has the same customization arguments as the hist function.

print('Bins = 30')
plt.hist2d(x=df['SalePrice'], y=df['GrLivArea'], cmap='Blues', bins=30)
plt.colorbar()
# I think this one is the best to read, the one with higher is too detailed and the one with lower
# doesn't capture the high amount of values on the right side of the distribution

# (a) Display on the same line in two different scatter plots, the sales
#  price as a function of each of these two variables with the names of the labels.
plt.scatter(df['TotalBsmtSf'], df['GrLivArea'])

plt.figure(figsize=[20, 10])

plt.subplot(121)

plt.scatter(df['SalePrice'], df['TotalBsmtSF'], label='Total Bsmt SF', color='red')
plt.xlabel('Sales Price')
plt.legend()

plt.subplot(122)

plt.scatter(df['SalePrice'], df['GrLivArea'], label='Gr Liv Area', color='purple')
plt.xlabel('Sales Price')
plt.legend()

# (b) What is your analysis regarding these two graphs? What appears to be
#  the relationship between the data ?
# The relationship appears to be positive between Sales Price and the two other variables, indicated by the scatter cloud
# in the left bottom corner.
# However, on both bivariate distribution we have extreme values that doesn't fit the pattern.
# nonetheless, it seems that not another (non-linear) relationship can describe the data better.


# (a) On the same graph, display the box plots of selling prices for each
#  quality level. For good visibility, feel free to initialize a large figure
#  with the figsize attribute of the plt.figure() method.
df.boxplot(column='SalePrice', by='OverallQual', figsize=[20, 10])
# (b) How would you explain the increasing disparity in data as a
# function of finish quality ?
# We can see that for low quality the price seems to be quite clear and doesn't have a high range. However, if we increase
# the quality, the price range gets wider and therefore, boxplots are wider. In addition, with increasing
# quality, there are extreme value on both ends, indicating a wide range of sales prices for higher quality.

#  Synthesis
# Finally, let's synthesize the previous studies into a single graph.
# (a) Reuse the graph of price versus living area. Color the points according
#  to the quality of finish of the house. Vary the size of the points according
#  to the area of the basement. We will use a coefficient of opacity of 0.4.
# (b) Add a box on the graph in which you explain which variable is
#  represented by the size of the samples and which variable is represented
#  by their color.
# Tips:
#  You can use the following figure as inspiration for your graph
#  (the rainbow cmap was used here).
# You can display the colorbar of the graph with the code: plt.colorbar().

plt.figure(figsize=[10, 10])
plt.scatter(
    df['GrLivArea'],
    df['SalePrice'],
    label='Gr Liv Area',
    cmap='rainbow',
    c=df['GrLivArea'],
    s=df['TotalBsmtSF'],
    alpha=0.4,
)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.colorbar()

# plt.axes([0.65, 0.65, 0.2, 0.15], facecolor = '#ffe5c1')

plt.plot([300, 1200, 1200, 300, 300], [68000, 68000, 74000, 74000, 6800], 'b')


plt.text(0.8, 1.1, 'Size: TotalBsmtSF')
plt.text(0.8, 1.1, 'Color: OverallQual')
