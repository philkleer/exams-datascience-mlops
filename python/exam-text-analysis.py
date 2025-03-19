# Text Mining

# Instructions

# The exercise is composed of several questions, do them in order and be careful to respect the names of the variables.

# The exam is composed of two sections: a first section where you will be asked to build a word cloud from the last speeches given by Donald Trump, which you will have previously cleaned up and formatted (use of regular expressions, tokenization, stop words filtering). In the second section, you will be asked to build a simple machine learning model to perform a sentiment analysis on the different paragraphs of these speeches.

# Explore our dataset and clean it. The dataset we will use contains excerpts from Donald Trump's speeches during his 2016 campaign.
# (a) Using the pandas package, import the data contained in the file "trump.csv" into a DataFrame named df using the following separator : '|'.
# (b) Display the first 10 rows of the dataset.

import pandas as pd

df = pd.read_csv('trump.csv', sep='|')

df.head(10)

# (c) Create a string named speeches containing the concatenation of all entries in the "Speech" column of df. Be sure to insert a space between each line.
# Take care, during the examination, not to display the global content of the speeches variable as this may take several minutes.
speeches = ""
for item in df.Speech:
    speeches += item + ' '

# In the text, the audience reactions are indicated by square brackets, [Applause] for example.

# (d) Create a test variable containing the string: Hello World! [Applause] How are you ?.
test = 'Hello World! [Applause] How are you?'

# (e) Using the re library, create a function named remove_brackets which takes as input a string and replaces the words in square brackets with spaces.

# (f) Apply this function to test to check its effectiveness.

import re

def remove_brackets(string):    
    r = re.compile(r"\s?\[.*?\]\s?")
    new_string = r.sub(' ', string)
    return new_string

remove_brackets(test)

# (g) Apply the remove_brackets function to the speeches string.
speeches_rm = remove_brackets(speeches)

# (h) Using the nltk library, create a variable named stop_words containing the common stop words of the English language and the following characters and words:
# ["?", "!", ".", ",", ":", ";", "-", "--", "...", "\"", "'", "they've", "they're", "they'll", "i've", "i'm", "i'll", "could"]

# In the character "\"" , the double quote " is preceded by a backslash \ to indicate it as part of the string within the list.

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

new_stop_words = ["?", "!", ".", ",", ":", ";", "-", "--", "...", "\"", "'", "they've", "they're", "they'll", "i've", "i'm", "i'll", "could"]

stop_words.update(new_stop_words)

# The word_tokenize function of the nltk.tokenize sub-module treats contractions ("we'll", "didn't") as two different words (Ex: "didn't" => ["did", "not"]), to avoid this we will use the TweetTokenizer class which does not make this distinction.

# (i) Import the TweetTokenizer class from the nltk.tokenize package.

# (j) Instantiate the TweetTokenizer class in a tokenizer object.

# (h) Apply the tokenize method of tokenizer to speeches and store the result in a variable named tokens.

from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

tokens = tokenizer.tokenize(speeches)

# (k) Display the total number of words as well as the number of different words found in these speeches.
len(set(tokens))

# (l) Remove all stop words from the tokens list.
def stop_words_filtering(words) : 
    tokens = []
    for word in words:
        if word not in stop_words:
            tokens.append(word)
    return tokens

tokens = stop_words_filtering(tokens)

# (m) Display the total number of words.
len(tokens)

# 2. Wordcloud construction

# In this section we will build a wordcloud from the tokens variable.

# (a) Import the WordCloud class from the wordcloud module.

# (b) Instantiate the word cloud layer wc from the WordCloud class, taking as parameters :

# A white background colour
# A maximum number of words to display equal to 1000.
# A maximum font size of 90.
# collocations set to False.
# random_state set to 42.
# mask set with the mask variable instantiated below.
# (c) Using the matplotlib.pyplot sub-module, display the wordcloud.
from wordcloud import WordCloud

# I copied this from another code chunk above, but a proper task is missing in regard of this
from PIL import Image
import numpy as np

mask = np.array(Image.open("trump.jpg"))

wc = WordCloud(
    background_color="white", 
    max_words=1000, 
    max_font_size=90, 
    collocations=False,
    mask=mask,
    random_state=42
)
# above i did create an array of tokens instead of a single string that is needed for the word cloud

tokens2 = ""

for token in tokens:
    tokens2 += token + ' '

# We want to customize the wordcloud, in particular to change the colours of the text, so that it automatically takes the colours of the original image.

# (d) Import the ImageColorGenerator class from the wordcloud module.

# (e) Instantiate an ImageColorGenerator object named img_color by specifying the mask variable previously created as the constructor argument.

# (f) Use the recolor method of the WordCloud class and give it as argument color_func=img_color.

# For more smoothing on the letters, use the "interpolation="bilinear" option in the imshow function of pyplot.
# (g) Remove the axes from your graph.

# (h) Display the wordcloud again.
from wordcloud import ImageColorGenerator
import matplotlib.pyplot as plt 

img_color = ImageColorGenerator(mask)

plt.figure(figsize= (10,5))
wc.generate(tokens2)
wc.recolor(color_func=img_color)
plt.imshow(wc, interpolation='bilinear')
plt.show()

# 3. Sentiment analysis

# In this part, we will try to create a sentiment analysis model on this dataset.

# (a) Import the train_test_split function from the sklearn.model_selection submodule.

# (b) Create a Series X containing the column "Speech" of df as well as a Series y containing the column "Sentiment".

# (c) Using the train_test_split function, create the X_train, X_test, y_train and y_test datasets. We will take a test set size equivalent to 25% of the total data set.

from sklearn.model_selection import train_test_split

X = df.Speech
y = df.Sentiment

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7786)

# The second step is vectorization, which consists of converting each paragraph of the various speeches into a numerical representation. This involves creating a corpus and a term-document matrix.

# (d) Import the CountVectorizer class from the sklearn.feature_extraction.text sub-module.

# (e) Instantiate the CountVectorizer class into an object named vectorizer.

# (f) Apply the fit_transform and transform methods of the vectorizer on X_train and X_test respectively.
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# We will now train a decision tree and a GradientBoostingClassifier model.

# (g) Train a decision tree named decision_tree on X_train with the default hyperparameters (sub-module sklearn.tree).

# (h) Train a GradientBoostingClassifier named gradient_boosting on X_train with the default hyperparameters (sub-module sklearn.ensemble).
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)

# (i) Using the accuracy on the test set, compare the performance of the two algorithms.
from sklearn.metrics import accuracy_score

y_pred_dt = dt.predict(X_test)
y_pred_gbc = gbc.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_gbc = accuracy_score(y_test, y_pred_gbc)

print(f'Decision Tree Accuracy: {accuracy_dt:.4f}')
print(f'Gradien Boosting Accuracy: {accuracy_gbc:.4f}')

# (j) Display the confusion matrices of these two algorithms on the test set. Which sentiment is difficult to detect ?
confusion_matrix_dt = pd.crosstab(y_test, y_pred_dt, rownames=['Actual class'], colnames=['Predicted class'])
confusion_matrix_gbc = pd.crosstab(y_test, y_pred_gbc, rownames=['Actual class'], colnames=['Predicted class'])

print('#'*10, ' Decision Tree ', '#'*10, '\n')
print(confusion_matrix_dt, end='\n\n\n')
print('#'*10, ' Gradient Boosting ', '#'*10, '\n')
print(confusion_matrix_gbc, end='\n\n\n')

# In both cases the one neutral tweet was not correctly assigned to neutral (DT: negative; GBC: positive)
# DT has also problems detecting negative and positive cases correctly, putting more or less
#  30% of the cases in the wrong class
# GBC is a bit better among negative cases (20% still wrong) than for positive cases (still 26% wrong)