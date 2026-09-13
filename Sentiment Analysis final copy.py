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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('covidvaccine.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


missed = pd.DataFrame()
missed['column'] = df.columns

missed['percent'] = [round(100* df[col].isnull().sum() / len(df), 2) for col in df.columns]
missed = missed.sort_values('percent',ascending=False)
missed = missed[missed['percent']>0]

fig = sns.barplot(
    x=missed['percent'], 
    y=missed["column"], 
    orientation='horizontal'
).set_title('Missed values percent for every column')


# In[8]:


def unique_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    tt['Percentage']=tt['Uniques']/tt['Total']
    return(np.transpose(tt))
unique_values(df)


# In[9]:


def most_frequent_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        itm = data[col].value_counts().index[0]
        val = data[col].value_counts().values[0]
        items.append(itm)
        vals.append(val)
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    return(np.transpose(tt))
most_frequent_values(df)


# In[10]:


ds = df['user_name'].value_counts().reset_index()
ds.columns = ['user_name', 'tweets_count']
ds = ds.sort_values(['tweets_count'],ascending=False)
df = pd.merge(df, ds, on='user_name')

fig = sns.barplot( 
    x=ds.head(20)["tweets_count"], 
    y=ds.head(20)["user_name"], 
    orientation='horizontal'
).set_title('Top 20 users by number of tweets')


# In[11]:


ds = df['source'].value_counts().reset_index()
ds.columns = ['source', 'count']
ds = ds.sort_values(['count'],ascending=False)

fig = sns.barplot(
    x=ds.head(10)["count"], 
    y=ds.head(10)["source"], 
    orientation='horizontal', 
    #title='Top 40 user sources by number of tweets', 
    #width=800, 
    #height=800
).set_title('Top 10 user sources by number of tweets')


# In[12]:


df['hour'] = pd.to_datetime(df['date'], errors='coerce').dt.hour
ds = df['hour'].value_counts().reset_index()
ds.columns = ['hour', 'count']
ds = ds.sort_values(by='hour')
hour_labels = {
    5: '5 AM', 6: '6 AM', 7: '7 AM', 8: '8 AM', 9: '9 AM', 10: '10 AM',
    11: '11 AM', 12: '12 PM', 13: '1 PM', 14: '2 PM', 15: '3 PM', 
    16: '4 PM', 17: '5 PM', 18: '6 PM', 19: '7 PM', 20: '8 PM', 
    21: '9 PM', 22: '10 PM', 23: '11 PM', 24: '12 AM', 1: '1 AM', 2:'2 AM', 3: '3 AM', 4: '4 AM'
}
ds['hour'] = ds['hour'].map(hour_labels)


# Plotting

plt.figure(figsize=(10, 6))
sns.barplot(x=ds["hour"], y=ds["count"], color='skyblue')
plt.title('Tweets distribution over hours')
plt.xlabel('Hour')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45)  # Rotate x-labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()




# In[13]:


df['Timestamp'] = pd.to_datetime(df.user_created, format="%d-%m-%Y %H:%M", errors='coerce')
mask = df.Timestamp.isnull()
df.loc[mask, 'Timestamp'] = pd.to_datetime(df[mask]['user_created'], format='%Y-%m-%d %H:%M:%S',
                                             errors='coerce')
df['year_created'] = df['Timestamp'].dt.year
# Keep rows with 'year_created' greater than 1970
data = df.drop_duplicates(subset='user_name', keep="first")
data = data[data['year_created'] > 1970]
# Count the occurrences of each year
data = data['year_created'].value_counts().reset_index()
data.columns = ['year', 'number']
# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x=data["year"], y=data["number"], color='skyblue')
plt.xlabel('Year Created')
plt.ylabel('Number of Users')
plt.title('User Creation Year Distribution')
# Adjust x-ticks for better spacing
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# In[14]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=50,
        max_font_size=40, 
        scale=5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(10,10))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)
    plt.imshow(wordcloud)
    plt.show()
    
    
    
    
    
    
india_df = df.loc[df.user_location=="India"]
show_wordcloud(india_df['text'], title = 'Prevalent words in tweets from India')




# In[15]:


us_df = df.loc[df.user_location=="United States"]
show_wordcloud(us_df['text'], title = 'Prevalent words in tweets from US')




# In[16]:


df.columns


# In[17]:


text_df = df.drop(['user_name', 'user_location', 'user_description', 'user_created',
       'user_followers', 'user_friends', 'user_favourites', 'user_verified',
       'date', 'hashtags', 'source', 'is_retweet'],axis=1)
text_df.head()


# In[18]:


print(text_df['text'].iloc[0],"\n")
print(text_df['text'].iloc[1],"\n")
print(text_df['text'].iloc[2],"\n")
print(text_df['text'].iloc[3],"\n")
print(text_df['text'].iloc[4],"\n")


# In[19]:


text_df.info()


# In[20]:


simple_text='This isn\'t a real text, this is an example text...Notice this contains punctuation!!'







# In[21]:


import pandas as pd
import nltk
import re


tokenizer = nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9\']+')
tokenized_document = tokenizer.tokenize(simple_text)
print(tokenized_document)









# In[22]:


stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)


# In[23]:


#remove stopwords
cleaned_tokens = []
for word in tokenized_document:
    word = word.lower()
    if word not in stop_words:
        cleaned_tokens.append(word)
print(cleaned_tokens)    




# In[24]:


# we can also remove stopwords using list comprehension
cleaned_tokens = [word.lower() for word in tokenized_document if word.lower() not in stop_words]
print(cleaned_tokens)



# In[25]:


# explore lemmatization vs stemming
lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()

words = ['cacti','sings','hopped','rocks','better','easily']
pos = ['n','v','v','n','a','r']
lemmatized_words = [lemmatizer.lemmatize(words[i], pos=pos[i]) for i in range(6)]
stemmed_words = [stemmer.stem(word) for word in words]

print(" Lemmatized words: ", lemmatized_words)
print("Stemmed words: ", stemmed_words)



# In[26]:


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Create a stemming object
stemmer = PorterStemmer()
cleaned_tokens = [word.lower() for word in tokenized_document if word.lower() not in stop_words]
print(cleaned_tokens)
# Perform stemming on the tokens
stemmed_text = [stemmer.stem(word) for word in cleaned_tokens]
print(stemmed_text)




# In[27]:


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


# In[28]:


text_df.head()


# In[29]:


print(text_df['text'].iloc[0],"\n")
print(text_df['text'].iloc[1],"\n")
print(text_df['text'].iloc[2],"\n")
print(text_df['text'].iloc[3],"\n")
print(text_df['text'].iloc[4],"\n")


# In[30]:


text_df.info()


# In[31]:


#Sentiment analysis 

def polarity(text):
    return TextBlob(text).sentiment.polarity
text_df['polarity'] = text_df['text'].apply(polarity)
text_df.head(5)


# In[32]:


def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label > 0:
        return "Positive"
text_df['sentiment'] = text_df['polarity'].apply(sentiment)
text_df.head()



# In[33]:


fig = plt.figure(figsize=(5,5))
sns.countplot(x='sentiment', data = text_df)






# In[34]:


fig = plt.figure(figsize=(7,7))
colors = ("yellowgreen", "gold", "red")
wp = {'linewidth':2, 'edgecolor':"black"}
tags = text_df["sentiment"].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,
         startangle=90, wedgeprops = wp, explode =explode, label='')
plt.title('Distribution of sentiments')


# In[35]:


# Import Counter module to count word frequency
from collections import Counter
# Separate text data into positive and negative sentiments
positive_texts = text_df[text_df['sentiment'] == 'Positive']['text']
negative_texts = text_df[text_df['sentiment'] == 'Negative']['text']
# Join all the positive and negative texts into single strings
positive_text = ' '.join(positive_texts)
negative_text = ' '.join(negative_texts)
# Count word frequency for positive and negative texts
positive_word_freq = Counter(positive_text.split())
negative_word_freq = Counter(negative_text.split())
# Get top 100 to 150 words for positive and negative sentiments
top_positive_words = [word for word, _ in positive_word_freq.most_common(150)]
top_negative_words = [word for word, _ in negative_word_freq.most_common(150)]
# Join top positive and negative words into single strings
top_positive_text = ' '.join(top_positive_words)
top_negative_text = ' '.join(top_negative_words)
# Generate word clouds for top positive and negative words
top_positive_wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=150).generate(top_positive_text)
top_negative_wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=150).generate(top_negative_text)

# Plot the word clouds
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(top_positive_wordcloud, interpolation='bilinear')
plt.title('Top Positive Words')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(top_negative_wordcloud, interpolation='bilinear')
plt.title('Top Negative Words')
plt.axis('off')
plt.show()


# In[36]:


def remove_tag(string):
    text=re.sub('<.*?>','',string)
    return text
def remove_mention(text):
    line=re.sub(r'@\w+','',text)
    return line
def remove_hash(text):
    line=re.sub(r'#\w+','',text)
    return line
def remove_newline(string):
    text=re.sub('\n','',string)
    return text
def remove_url(string): 
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',string)
    return text
def remove_number(text):
    line=re.sub(r'[0-9]+','',text)
    return line
def remove_punct(text):
    line = re.sub(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*','',text)
    return line
def text_strip(string):
    line=re.sub('\s{2,}', ' ', string.strip())
    return line
def remove_thi_amp_ha_words(string):
    line=re.sub(r'\bamp\b|\bthi\b|\bha\b',' ',string)
    return line
df['refine_text']=df['text'].str.lower()
df['refine_text']=df['refine_text'].apply(lambda x:remove_tag(str(x)))
df['refine_text']=df['refine_text'].apply(lambda x:remove_mention(str(x)))
df['refine_text']=df['refine_text'].apply(lambda x:remove_hash(str(x)))
df['refine_text']=df['refine_text'].apply(lambda x:remove_newline(x))
df['refine_text']=df['refine_text'].apply(lambda x:remove_url(x))
df['refine_text']=df['refine_text'].apply(lambda x:remove_number(x))
df['refine_text']=df['refine_text'].apply(lambda x:remove_punct(x))
df['refine_text']=df['refine_text'].apply(lambda x:remove_thi_amp_ha_words(x))
df['refine_text']=df['refine_text'].apply(lambda x:text_strip(x))
df['text_length']=df['refine_text'].str.split().map(lambda x: len(x))


# In[37]:


import plotly.graph_objects as go

fig = go.Figure(data=go.Violin(y=df['text_length'], box_visible=True, line_color='black',
                               meanline_visible=True, fillcolor='royalblue', opacity=0.6,
                               x0='Tweet Text Length'))

fig.update_layout(yaxis_zeroline=False,title="Distribution of Text length",template='ggplot2')
fig.show()


# In[38]:


from plotly.subplots import make_subplots

def ngram_df(corpus,nrange,n=None):
    vec = CountVectorizer(stop_words = 'english',ngram_range=nrange).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    total_list=words_freq[:n]
    df=pd.DataFrame(total_list,columns=['text','count'])
    return df
unigram_df=ngram_df(df['refine_text'],(1,1),20)
bigram_df=ngram_df(df['refine_text'],(2,2),20)
trigram_df=ngram_df(df['refine_text'],(3,3),20)

fig = make_subplots(
    rows=3, cols=1,subplot_titles=("Unigram","Bigram",'Trigram'),
    specs=[[{"type": "scatter"}],
           [{"type": "scatter"}],
           [{"type": "scatter"}]
          ])

fig.add_trace(go.Bar(
    y=unigram_df['text'][::-1],
    x=unigram_df['count'][::-1],
    marker={'color': "blue"},  
    text=unigram_df['count'],
    textposition = "outside",
    orientation="h",
    name="Months",
),row=1,col=1)

fig.add_trace(go.Bar(
    y=bigram_df['text'][::-1],
    x=bigram_df['count'][::-1],
    marker={'color': "blue"},  
    text=bigram_df['count'],
     name="Days",
    textposition = "outside",
    orientation="h",
),row=2,col=1)

fig.add_trace(go.Bar(
    y=trigram_df['text'][::-1],
    x=trigram_df['count'][::-1],
    marker={'color': "blue"},  
    text=trigram_df['count'],
     name="Days",
    orientation="h",
    textposition = "outside",
),row=3,col=1)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='Top N Grams',xaxis_title=" ",yaxis_title=" ",
                  showlegend=False,title_x=0.5,height=1200,template="plotly_dark")
fig.show()


# In[39]:


# Tweet Length Analysis
text_df['tweet_length'] = text_df['text'].apply(len)  # Compute tweet length
plt.figure(figsize=(8, 6))
sns.histplot(text_df['tweet_length'], bins=30, kde=True)
plt.title('Distribution of Tweet Length')
plt.xlabel('Tweet Length (characters)')
plt.ylabel('Frequency')
plt.show()


# In[40]:


vect = CountVectorizer(ngram_range=(1,2)).fit(text_df['text'])
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd  # Make sure to import pandas module

# Instantiate CountVectorizer object with desired ngram_range
vect = CountVectorizer(ngram_range=(1, 2))

# Fit CountVectorizer to your text data and transform it (sparse matrix)
X_sparse = vect.fit_transform(text_df['text'])

# Work directly with the sparse matrix representation
print(X_sparse.shape)  # Print the shape of the sparse matrix



# In[41]:


X = text_df['text']
Y = text_df['sentiment']

X = vect.transform(X)

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of x_test:", (x_test.shape))
print("Size of y_test:", (y_test.shape))



# In[42]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
sns.countplot(x=y_train_balanced)




# In[43]:


from sklearn.feature_selection import VarianceThreshold
variance_selector = VarianceThreshold(threshold=0)
x_train_selected = variance_selector.fit_transform(x_train_balanced)
x_test_selected = variance_selector.transform(x_test)
print(f"{x_train_balanced.shape[1] - x_train_selected.shape[1]} features have been removed, {x_train_selected.shape[1]} features remain")


# In[44]:


selected_features = variance_selector.get_support()
print("Size of boolean array:", selected_features.size)


# In[45]:


import seaborn as sns
import numpy as np
# Convert the boolean array to a one-dimensional numpy array
selected_features_array = np.array(selected_features)
# Reshape the array to have a single row
selected_features_row = selected_features_array.reshape(1, -1)
# Plot the boolean array as a heatmap
plt.figure(figsize=(15, 2))
sns.heatmap(selected_features_row, cmap='rocket', cbar=False)
plt.xlabel('Feature Index')
plt.ylabel('Feature Selection')
plt.title('Feature Selection: Variance Threshold')
plt.show()


# In[46]:


from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=200)  # You can adjust the value of k as needed
x_train_selected = selector.fit_transform(x_train_balanced, y_train_balanced)
x_test_selected = selector.transform(x_test)

# Print the shape of the transformed training and testing data
print("Shape of x_train_selected:", x_train_selected.shape)
print("Shape of x_test_selected:", x_test_selected.shape)

selected_feature_indices = selector.get_support(indices=True)
print("Indices of selected features:", selected_feature_indices)


# In[47]:


import matplotlib.pyplot as plt
import seaborn as sns
# Get the indices of features retained by VarianceThreshold
variance_threshold_indices = variance_selector.get_support(indices=True)
# Create a Boolean array indicating the features selected by SelectKBest
selected_features_boolean = np.zeros(x_train_balanced.shape[1], dtype=bool)
selected_features_boolean[selected_feature_indices] = True

# Create a Boolean array indicating features retained by VarianceThreshold
variance_threshold_boolean = np.zeros(x_train_balanced.shape[1], dtype=bool)
variance_threshold_boolean[variance_threshold_indices] = True

# Combine the two Boolean arrays to get features removed by both selectors
removed_features_boolean = ~(selected_features_boolean | variance_threshold_boolean)

# Reshape the array for visualization
removed_features_reshaped = removed_features_boolean.reshape(1, -1)

# Plot the heatmap of removed features
plt.figure(figsize=(15, 2))
sns.heatmap(removed_features_reshaped, cmap='rocket', cbar=False)
plt.xlabel('Feature Index')
plt.ylabel('Feature Selection')
plt.title('Features Removed by VarianceThreshold and SelectKBest')
plt.show()


# In[48]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
x_train_scaled = scaler.fit_transform(x_train_selected)
x_test_scaled = scaler.transform(x_test_selected)









# In[49]:


from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix


# In[50]:


from sklearn.linear_model import LogisticRegression

# Define the hyperparameters grid for tuning
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

# Initialize Logistic Regression classifier
logistic_regression = LogisticRegression(max_iter=1000, random_state=42)

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(logistic_regression, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train_scaled, y_train_balanced)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)





# In[51]:


# Train the Logistic Regression classifier with the best hyperparameters
best_logistic_regression = LogisticRegression(**best_params, max_iter=1000, random_state=42)
best_logistic_regression.fit(x_train_scaled, y_train_balanced)

# Predict the sentiment labels for the test data
y_pred_logistic = best_logistic_regression.predict(x_test_scaled)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred_logistic)
print("Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_logistic)
print("Confusion Matrix:\n", conf_matrix)

# Generate classification report
print("Classification Report:\n", classification_report(y_test, y_pred_logistic))




# In[52]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predict the sentiment labels for the test data
y_pred_logistic = best_logistic_regression.predict(x_test_scaled)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_logistic)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[53]:


from sklearn.naive_bayes import MultinomialNB

# Define the hyperparameters grid for tuning
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0]  # Smoothing parameter
}

# Initialize Multinomial Naive Bayes classifier
mnb_classifier = MultinomialNB()

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(mnb_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train_scaled, y_train_balanced)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)





# In[54]:


# Train the Multinomial Naive Bayes classifier with the best hyperparameters
best_mnb_classifier = MultinomialNB(**best_params)
best_mnb_classifier.fit(x_train_scaled, y_train_balanced)

# Predict the sentiment labels for the test data
y_pred_mnb = best_mnb_classifier.predict(x_test_scaled)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred_mnb)
print("Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_mnb)
print("Confusion Matrix:\n", conf_matrix)

# Generate classification report
print("Classification Report:\n", classification_report(y_test, y_pred_mnb))


# In[55]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predict the sentiment labels for the test data
y_pred_mnb = best_mnb_classifier.predict(x_test_scaled)

# Compute confusion matrix
conf_matrix_mnb = confusion_matrix(y_test, y_pred_mnb)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_mnb, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:




