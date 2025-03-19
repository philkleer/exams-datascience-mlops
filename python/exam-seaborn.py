# (a) Load the pandas package as pd, the seaborn package as sns and the sub-package matplotlib.pyplot as plt.
# (b) Import the file 'netflix_titles.csv' into a DataFrame called df_netflix.
# (c) Display the first 5 contents of the Netflix catalog.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_netflix = pd.read_csv('sprint2/data/netflix_titles.csv')

df_netflix.head()

# (d) Display, in a graph, the number of contents belonging to each of the
#  two types ('TV Show' and 'Movie').
sns.countplot(data=df_netflix, x='type')
plt.title('Netflix content by type')
plt.xlabel('Type')
plt.ylabel('Count')

# Other solution by wrangling data before
# tv_types = df_netflix['type'].value_counts()
# print(tv_types)
# sns.barplot(x = tv_types.index, y = tv_types.values)

# (e) Now import the file `'imdb.csv'` into a `DataFrame` called `df_imdb` and
#  display the first 5 lines.
df_imdb = pd.read_csv('sprint2/data/imdb.csv')
df_imdb.head()

# Rename columsn so that they are the same as in the exam
df_imdb = df_imdb.rename(
    {
        'Series_Title': 'primaryTitle',
        'Released_Year': 'startYear',
        'IMDB_Rating': 'averageRating',
        'No_of_Votes': 'numVotes',
    },
    axis=1,
)

df_imdb.loc[df_imdb['startYear'] == 'PG', 'startYear'] = np.nan

df_imdb = df_imdb.dropna(subset='startYear')

df_imdb['startYear'] = df_imdb['startYear'].astype('int')


# (f) To facilitate the analysis of the films of the catalog, merge the
#  variables df_netflix and df_imdb in a DataFrame all_content,
#  in keeping the lines where the title of the content on Netflix and
#  its release year is the same as the main title and release
#  year according to IMDB.
all_content = pd.merge(
    df_netflix,
    df_imdb,
    left_on=['title', 'release_year'],
    right_on=['primaryTitle', 'startYear'],
    how='inner',
)

# (g) Add to `all_content` an `'is_action'` column which will take the
#  value `True` if a content belongs to the *Action* category, and
#  `False` otherwise.
# > As a reminder, the list of categories to which a film belongs
#  is available in the `'listed_in'` variable. You can call the `apply`
#  method on this variable by specifying a relevant `lambda` function.
all_content['is_action'] = all_content['listed_in'].apply(lambda x: 'Action' in x)

# Just to test if it is right
print(all_content[['is_action', 'listed_in']].head())

# (h) Extract from all_content a DataFrame named all_movies,
#  containing only movies (contents of type 'Movie').
# (i) Create, from the 'duration' variable, a new 'duration_int'
#  variable in all_movies, where the last four characters will
#  be truncated. Convert the type of the 'duration_int' variable to int.
all_movies = all_content.loc[(all_content['type'] == 'Movie')]

# contains only Movies
all_movies['type'].unique()

# We need to slice from the 4th last element (' min').
all_movies['duration_int'] = all_movies.duration.str[:-4]

# Manipulate to type integer
all_movies['duration_int'] = all_movies['duration_int'].astype('int')

# Check if type changed (due to error messages)
all_movies.info()

# (j) Display, in a `boxplot`, the distribution of the
#  `'duration_int'` variable depending on whether the film belongs
#  to the *Action* category or not.
sns.catplot(kind='box', x='is_action', y='duration_int', data=all_movies)

plt.title('Duration in dependence of movie being Action or not')
plt.xlabel('Film is action?')
plt.ylabel('Duration in minutes')


# (k) Display a histogram accompanied by a density estimation curve in
#  order to analyze the `'averageRating'` variable.
plt.figure()

# old distplot
sns.displot(all_movies['averageRating'], kde=True)
plt.title('Average rating of movies on Netflix')
plt.xlabel('Average Rating (0-10)')

# warning to change to distplot but it is plotted correctly

# (l) Display, using a curve, the relationship between the duration of an
#  action movie whose 'country' variable is 'United States' and its average
#  rating on IMDB. We will only consider films with a duration of
#  less than 160 minutes.
# Checking variable country
all_movies['country'].value_counts()

all_action_us = all_movies.loc[
    (all_movies['country'] == 'United States')
    & (all_movies['is_action'])
    & (all_movies['duration_int'] < 160)
]

# Building scatter plot
# The comment on my solution said that a line plot would be more appropriate
# I disagree on this, I think a scatter plot makes more sense, since we averageRating
# and duration_int have duplicated entries, and therefore a lineplot will not be nice
# (you can see it by the lightblue line which is due to datapoints very close to each other)
# (I think a lineplot makes sense, when one variable has unique values that
#  are not duplicated, so that you can make a nice order for the line)
# But here is the solution that was demanded.
plt.figure()
sns.lineplot(data=all_action_us, x='duration_int', y='averageRating')
plt.title('Relationship between duration and rating for Action Movies from the US')
plt.xlabel('Duration in Minutes')
plt.ylabel('Average Rating')

sns.lmplot(data=all_action_us, x='duration_int', y='averageRating', lowess=True)
plt.title('Relationship between duration and rating for Action Movies from the US')
plt.xlabel('Duration in Minutes')
plt.ylabel('Average Rating')

# Includes string that indicates more than one country. However task says that
# we just take the string 'United States', not the presence of United States in
# the string.
# Therefore we create a variable that test if United States is in the string
# Alternative solution with United States in String:
all_movies['is_usa'] = all_movies['country'].apply(lambda x: 'United States' in x)

all_action_us2 = all_movies.loc[
    (all_movies['is_action'])
    & (all_movies['is_usa'])
    & (all_movies['duration_int'] < 160)
]

# Building scatter plot
plt.figure()
sns.scatterplot(data=all_action_us2, x='duration_int', y='averageRating')
plt.title('Relationship between duration and rating for Action Movies from the US')
plt.xlabel('Duration in Minutes')
plt.ylabel('Average Rating')

# The 'director' column contains the director(s) of the Netflix content,
#  separated by the characters ', '.
# The method str.split() applied to a Series with the parameter expand=True,
#  allows to recover a DataFrame with for each line the character strings separated
#  by the separator indicated in a different column.
# The stack() method is used to transform a DataFrame into a Serie Multi-index,
#  by stacking the (non-missing) values of the columns one after the other.
#  So, for example:
# S = pd.Series(['hello friend', 'hello world', 'hi'])
# S.str.split(' ', expand=True).stack().reset_index(drop=True)
# allows to return the series containing the words: 'hello',
#  'friend', 'hello', 'world', 'hi'.

# (m) Store, in a Series named directors, all the directors present in the
#  column 'director' of all_content .
directors = (
    all_content['director'].str.split(', ', expand=True).stack().reset_index(drop=True)
)

directors

# (n) Display in a graph in horizontal bars, the 8 most present directors
#  in the catalog. We will use the attributes of `directors` for this.
most_present_directors = directors.value_counts().head(8)

sns.barplot(x=most_present_directors.index, y=most_present_directors.values)
plt.title('Most present directors')
plt.xlabel('Director')
plt.ylabel('Count')
plt.xticks(rotation=45, horizontalalignment='right')

# (o) Add to the DataFrame all_content a new variable 'year'
# containing the year in which content was added to the platform.
# (p) Display in a graph the evolution curve of the number of contents added
#  to the Netflix catalog. We will differentiate the type of content.
all_content['year'] = pd.to_datetime(all_content['date_added']).dt.year

all_content.head()

# First grouping by type
content_develop = all_content.groupby(['year', 'type']).size()

content_develop

# Plotting

sns.lineplot(data=content_develop, x='year', y=content_develop.values, hue='type')

plt.title('Development of added content by content type')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()

# (r) Store in two variables uk_movies and uk_series all films and series
#  of British origin. To do this, we will retrieve the elements of
#  all_content using loc.
# (s) It is accepted that the number of votes for a series or a film is a
#  reliable indicator of its popularity. Sort the two DataFrames in descending
#  order based on numVotes.
# (t) Display two graphs side by side containing the top 5 most popular English
#  series and the top 5 most popular English films.

# since it is not further declared, I do the same as above and filter only
#  by UK and not by UK + others.
uk_movies = all_content[
    (all_content['country'] == 'United Kingdom') & (all_content['type'] == 'Movie')
]

uk_series = all_content.loc[
    (all_content['country'] == 'United Kingdom') & (all_content['type'] == 'TV Show')
]

# NEED TO ADJUST!!! to numVotes!!!!
uk_movies = uk_movies.sort_values(by='numVotes', ascending=False)

uk_series = uk_series.sort_values(by='numVotes', ascending=False)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=[10, 10], sharey=True)

sns.barplot(ax=ax[0], data=uk_movies.head(5), x='title', y='numVotes')
sns.barplot(ax=ax[1], data=uk_series.head(5), x='title', y='numVotes')

fig.title('Best rated TV shows and Movies from the UK')

# Adjusting left plot
ax[0].set_xlabel('Movies')
ax[0].set_ylabel('Number of Votes')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right')

# Adjusting right plot
ax[1].set_xlabel('TV Shows')
ax[1].set_ylabel('Number of Votes')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right')

# (u) Using a graph, analyze the evolution of the quality of catalog
#  content over time.
# Locally it is release-year
quality_time = all_content.groupby('year')['averageRating'].mean()

quality_time

sns.lineplot(data=quality_time, x='year', y=quality_time.values)

plt.title('Development of average Rating over year (Movies + TV Shows)')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.ylim([0, 10])

# (v) Display the Top 15 most present actors/actresses in English productions
#  with an average rating above 8.
all_uk = all_content.loc[
    (all_content['country'] == 'United Kingdom') & (all_content['averageRating'] > 8)
]

actors_uk = all_uk['cast'].str.split(', ', expand=True).stack().reset_index(drop=True)

actors_uk

count_actors_uk = actors_uk.value_counts().head(15)

sns.barplot(x=count_actors_uk.index, y=count_actors_uk.values)
plt.title('Most present actors in UK productions')
plt.xlabel('Actor/Actress')
plt.ylabel('Count')
plt.xticks(rotation=45, horizontalalignment='right')
