import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # WordNetLemmatizer

# import seaborn as sns
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("data/train.csv")

# num_of_insults = df["Insult"].value_counts()

df_1 = df.dropna()
filtered_df = df_1.query("Insult!=0")


# text = filtered_df["Comment"].to_string(index=False)
# processed_text = re.sub(r"\bxa0\b", "", text)


# ### 3.4. Data transformation

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


# ## 4. Modeling
