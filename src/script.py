import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re


df = pd.read_csv("../data/train.csv")

### 2.2. Data exploration

df.head(20)

### 2.2.1 Columns description

# - insults: two possible outputs (0, 1). 0 means no insults, 1 means it contains insults (based on the presence of insult words, not accurate)
# - date: the date of the comment
# - comment: the comment being analyzed

num = df["Insult"].value_counts()

print(num)

plot = pd.DataFrame({"Comments": ["Insults", "No Insults"], "val": [num[0], num[1]]})
ax = plot.plot.bar(title="Number of insults", x="Comments", y="val", rot=0)

## 3. Data preparation and pre-processing

### 3.1. Data cleaning

#### Clean the data, only show data with insults

df_1 = df.dropna()
filtered_df = df_1.query("Insult!=0")
filtered_df.head(20)

### 3.2. Data visualization

#### Visualise the data in a wordcloud


text = filtered_df["Comment"].to_string(index=False)
processed_text = re.sub(r"\bxa0\b", "", text)

wordcloud = WordCloud(max_words=25, min_word_length=3).generate(processed_text)

plt.imshow(wordcloud, interpolation="bilinear")
plt.show()

# Create a document-term matrix
vectorizer = CountVectorizer()
dtm = vectorizer.fit_transform(reduced_data["Comment"])
vocab = vectorizer.get_feature_names()
df = pd.DataFrame(dtm.toarray(), columns=vocab)

# Compute the correlation matrix
# corr_matrix = df.corr()

# Create the heatmap
# sns.heatmap(corr_matrix, cmap="coolwarm")

### 3.3 Data reduction
### using random sampling

# reduced_data = filtered_df.sample(100)
# reduced_data.head(20)

### 3.4. Data transformation

# Download the required resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Tokenize the text

token_arr = []

for text in filtered_df["Comment"].values:
    tokens = word_tokenize(text)

    # Lowercase the tokens
    tokens = [token.lower() for token in tokens]

    tokens = [re.sub(r"[^a-zA-Z0-9]", "", token) for token in tokens]
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Stem the tokens
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = list(filter(None, tokens))
    token_arr.append(tokens)
# Print the processed tokens
print(token_arr)

### 3.5. Feature selection
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_df["Comment"])
print(tfidf_matrix)

text = filtered_df["Comment"]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(text)

feature_names = vectorizer.get_feature_names()
for i in range(len(text)):
    print("Document {}".format(i + 1))
    for j in range(len(feature_names)):
        print("{}: {}".format(feature_names[j], tfidf_matrix[i, j]))

## 4. Modeling
