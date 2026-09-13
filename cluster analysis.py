#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('covidvaccine.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.columns


# In[8]:


text_df = df.drop(['user_name', 'user_location', 'user_description', 'user_created',
       'user_followers', 'user_friends', 'user_favourites', 'user_verified',
       'date', 'hashtags', 'source', 'is_retweet'],axis=1)
text_df.head()


# In[9]:


simple_text='This isn\'t a real text, this is an example text...Notice this contains punctuation!!'







# In[10]:


import pandas as pd
import nltk
import re


tokenizer = nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9\']+')
tokenized_document = tokenizer.tokenize(simple_text)
print(tokenized_document)









# In[11]:


stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)


# In[12]:


#remove stopwords
cleaned_tokens = []
for word in tokenized_document:
    word = word.lower()
    if word not in stop_words:
        cleaned_tokens.append(word)
print(cleaned_tokens)    




# In[13]:


# we can also remove stopwords using list comprehension
cleaned_tokens = [word.lower() for word in tokenized_document if word.lower() not in stop_words]
print(cleaned_tokens)




# explore lemmatization vs stemming
lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()

words = ['cacti','sings','hopped','rocks','better','easily']
pos = ['n','v','v','n','a','r']
lemmatized_words = [lemmatizer.lemmatize(words[i], pos=pos[i]) for i in range(6)]
stemmed_words = [stemmer.stem(word) for word in words]

print(" Lemmatized words: ", lemmatized_words)
print("Stemmed words: ", stemmed_words)




# In[14]:


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Create a stemming object
stemmer = PorterStemmer()
cleaned_tokens = [word.lower() for word in tokenized_document if word.lower() not in stop_words]
print(cleaned_tokens)
# Perform stemming on the tokens
stemmed_text = [stemmer.stem(word) for word in cleaned_tokens]
print(stemmed_text)




# In[15]:


import pandas as pd
import nltk
import re
# Tokenization and Stopword Removal
tokenizer = nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9\']+')
stop_words = set(nltk.corpus.stopwords.words('english'))
# Stemmer
stemmer = nltk.stem.PorterStemmer()
def data_processing(text):
    if not isinstance(text, str):
        text = str(text)  # Convert to string if it's not already one
    text = text.lower()
    text = re.sub(r'https?\S+|www\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+|#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text
def stemming(text):
    # Tokenize
    tokenized_document = tokenizer.tokenize(text)
    # Remove stopwords
    cleaned_tokens = [word.lower() for word in tokenized_document if word.lower() not in stop_words]
    # Stemming
    stemmed_text = [stemmer.stem(word) for word in cleaned_tokens]
    return ' '.join(stemmed_text)
# Assuming 'text_df' is your DataFrame with the 'text' column
text_df['text'] = text_df['text'].apply(data_processing)
text_df = text_df.drop_duplicates('text')
text_df['text'] = text_df['text'].apply(lambda x: stemming(x))


# In[16]:


text_df.head()


# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer



# In[18]:


# Step 1: Vectorize Text Data
tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
text_tfidf = tfidf_vectorizer.fit_transform(text_df['text'])








# In[19]:


# Step 2: Apply Clustering Algorithm
num_clusters = 5  # You can adjust this parameter
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
pipeline = make_pipeline(Normalizer(), kmeans)
pipeline.fit(text_tfidf)





# In[20]:


from sklearn.decomposition import TruncatedSVD
# Initialize PCA with 2 components to reduce the dimensionality to 2
pca = TruncatedSVD(n_components=2)
# Apply PCA to the TF-IDF transformed text data
text_tfidf_pca = pca.fit_transform(text_tfidf)








# In[21]:


# Plot the PCA-transformed data
plt.figure(figsize=(10, 8))
plt.scatter(text_tfidf_pca[:, 0], text_tfidf_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('Clusters of Text Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.show()


# In[22]:


from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# Reduce dimensionality
svd = TruncatedSVD(n_components=100)
text_tfidf_reduced = svd.fit_transform(text_tfidf)

# Fit KMeans model
kmeans.fit(text_tfidf_reduced)



# Evaluate clustering performance
silhouette = silhouette_score(text_tfidf_reduced, kmeans.labels_)
davies_bouldin = davies_bouldin_score(text_tfidf_reduced, kmeans.labels_)
print("Silhouette Score:", silhouette)
print("Davies-Bouldin Index:", davies_bouldin)


# In[23]:


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.coherencemodel import CoherenceModel


# In[24]:


#Research Question for Cluster Analysis:

# How do the topics of discussion related to COVID-19 vaccination evolve over time, and what are the underlying patterns and trends in public discourse?

#unique insight
#topic clustering

# Apply Latent Dirichlet Allocation (LDA)
tf_vectorizer = CountVectorizer(max_df=0.5, min_df=2, stop_words='english')
text_tf = tf_vectorizer.fit_transform(text_df['text'])


# In[25]:


num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(text_tf)







# In[36]:


# Compute perplexity
perplexity = lda.perplexity(text_tf)
print("Perplexity:", perplexity)







# In[38]:


def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

num_top_words = 10
print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names_out()
display_topics(lda, tf_feature_names, num_top_words)





# In[39]:


# Define the most frequent words of interest
most_freq_words = ['health', 'vaccine', 'covid', 'covidshield', 'pfizer']
# Function to parse dates
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str)
    except (ValueError, TypeError):
        return pd.NaT  # Return NaT (Not a Time) for invalid dates or NoneType
# Apply the custom function to parse dates
df['date'] = df['date'].apply(parse_date)
# Drop rows with invalid dates
df = df.dropna(subset=['date'])
# Fill NaN values in the 'text' column with empty string
df['text'] = df['text'].fillna('')
# Create a DataFrame to store the frequency of words over time
word_freq_over_time = pd.DataFrame()
# Iterate through the most frequent words
for word in most_freq_words:
    # Initialize a list to store the frequency of the word over time
    freq_counter = []
    # Iterate through the text data to count occurrences of the word
    for text in df['text']:
        freq_counter.append(text.lower().count(word))  # Count occurrences of the word
    # Add the frequency list as a column in the DataFrame
    word_freq_over_time[word] = freq_counter
# Group the data by date and sum the frequency counts for each word
df_grouped = word_freq_over_time.groupby(df['date'].dt.date).sum()
# Plot the frequency of most frequent words over time
plt.figure(figsize=(12, 6))
for word in most_freq_words:
    plt.plot(df_grouped.index, df_grouped[word], label=word)
plt.title('Evolution of Most Frequent Words Over Time (All Dates)')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[40]:


import plotly.express as px  # Import Plotly Express library
# Group the data by date and sum the frequency counts for each word
df_grouped = word_freq_over_time.groupby(df['date'].dt.date).sum().reset_index()

# Melt the DataFrame to long format for plotting
df_melted = df_grouped.melt(id_vars='date', value_vars=most_freq_words, var_name='Word', value_name='Frequency')

# Plot the frequency of most frequent words over time using Plotly Express
fig = px.line(df_melted, x='date', y='Frequency', color='Word', title='Evolution of Most Frequent Words Over Time (2020-2022)')
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Frequency')
fig.show()

