#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
nltk.download(['stopwords','punkt','wordnet','omw-1.4','vader_lexicon'])
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


df = pd.read_csv('covidvaccine.csv')  # Replace 'your_dataset.csv' with the path to your dataset file


# In[3]:


df.head()


# In[4]:


print(df.info())


# In[5]:


print(df.describe())


# In[6]:


df.isnull().sum()


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


text_df = df.drop(['user_name', 'user_location', 'user_description', 'user_created',
       'user_followers', 'user_friends', 'user_favourites', 'user_verified',
       'date', 'hashtags', 'source', 'is_retweet'],axis=1)
text_df.head()


# In[10]:


def data_processing(text):
    if not isinstance(text, str):
        text = str(text)  # Convert to string if it's not already one
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','',text)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)
text_df.text = text_df['text'].apply(data_processing)
text_df = text_df.drop_duplicates('text')


# In[11]:


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data
text_df['text'] = text_df['text'].apply(lambda x: stemming(x))
text_df.head()


# In[12]:


def polarity(text):
    return TextBlob(text).sentiment.polarity
text_df['polarity'] = text_df['text'].apply(polarity)
text_df.head(5)


# In[13]:


def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label > 0:
        return "Positive"
text_df['sentiment'] = text_df['polarity'].apply(sentiment)
text_df.head()


# In[14]:


# Extract mentions from tweets
df['mentions'] = df['text'].str.findall(r'@(\w+)')

# Count the number of mentions per user
mention_counts = df['mentions'].explode().value_counts()

# Print the top users mentioned
print("Top Users Mentioned:")
print(mention_counts.head(10))

# Extract replies from tweets
df['replies'] = df['text'].str.contains(r'@(\w+)')

# Count the number of replies
reply_counts = df['replies'].sum()
print("Total Replies:", reply_counts)



# In[15]:


# Hashtag Analysis
def split_hashtags(x): 
    return str(x).replace('[', '').replace(']', '').split(',')
df = df.copy()
df['hashtag'] = df['hashtags'].apply(lambda row : split_hashtags(row))
df = df.explode('hashtag')
df['hashtag'] = df['hashtag'].astype(str).str.lower().str.replace("'", '').str.replace(" ", '')
df.loc[df['hashtag']=='', 'hashtag'] = 'NO HASHTAG'
ds = df['hashtag'].value_counts().reset_index()
ds.columns = ['hashtag', 'count']
ds = ds.sort_values(['count'],ascending=False)
fig = sns.barplot(
    x=ds.head(10)["count"], 
    y=ds.head(10)['hashtag'], 
    orientation='horizontal', 
).set_title('Top 10 hashtags')


# In[16]:


# Hypothesis Testing 


# Example: Hypothesis test for difference in user_followers between verified and non-verified users
verified_followers = df[df['user_verified'] == True]['user_followers']
non_verified_followers = df[df['user_verified'] == False]['user_followers']

from scipy import stats

t_stat, p_value = stats.ttest_ind(verified_followers, non_verified_followers, equal_var=False)
print("Hypothesis Test Results:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")





# In[17]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame containing 'date' and 'user_followers' columns

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



# Plot time series of user_followers
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['user_followers'], marker='o', linestyle='-')
plt.title('Time Series of User Followers (Entire Range)')
plt.xlabel('Date')
plt.ylabel('User Followers')
plt.grid(True)
plt.show()




# In[19]:


# Distribution of user followers
plt.figure(figsize=(10, 6))
sns.histplot(df['user_followers'], bins=30, kde=True)
plt.title('Distribution of User Followers')
plt.xlabel('Number of Followers')
plt.ylabel('Frequency')
plt.show()


# In[20]:


df['user_friends'] = pd.to_numeric(df['user_friends'], errors='coerce')  # 'coerce' will handle non-numeric values gracefully
df.dropna(subset=['user_friends'], inplace=True)
# Distribution of user friends using KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(df['user_friends'], shade=True)
plt.title('Distribution of User Friends')
plt.xlabel('Number of Friends')
plt.ylabel('Density')
plt.show()


# In[21]:


# Visualize distribution of followers and friends
plt.figure(figsize=(12, 6))
sns.histplot(df['user_followers'], bins=50, kde=True, color='blue', label='Followers')
sns.histplot(df['user_friends'], bins=50, kde=True, color='orange', label='Friends')
plt.title('Distribution of Followers and Friends')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[22]:


from textblob import TextBlob

# Assuming 'text' column contains the text data
# Calculate sentiment scores

df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Analyze Variation of Sentiment Among Different Demographic Groups

# Let's analyze sentiment variation among verified and non-verified users
plt.figure(figsize=(8, 6))
sns.boxplot(x='user_verified', y='sentiment', data=df)
plt.title('Sentiment Variation Between Verified and Non-Verified Users')
plt.xlabel('User Verified Status')
plt.ylabel('Sentiment Score')
plt.show()


# In[23]:


# Sentiment by user follower count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='user_followers', y='sentiment', data=df, alpha=0.5)
plt.title('Sentiment by User Follower Count')
plt.xlabel('Number of Followers')
plt.ylabel('Sentiment Score')
plt.show()


# In[24]:


# Analyze sentiment variation among users with varying follower counts
# For example, we can categorize users into groups based on follower count quartiles
df['follower_group'] = pd.qcut(df['user_followers'], q=4, duplicates='drop')  # Drop duplicate edges
plt.figure(figsize=(12, 6))
sns.boxplot(x='follower_group', y='sentiment', data=df)
plt.title('Sentiment Variation Among Users with Varying Follower Counts')
plt.xlabel('Follower Count Quartile')
plt.ylabel('Sentiment Score')
plt.xticks(rotation=45)
plt.show()


# In[25]:


# 3. Visualize Sentiment Scores Across Demographic Groups
# Replace this with your actual list of country names
all_countries = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia",
    "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium",
    "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad",
    "Chile", "China", "Colombia", "Comoros", "Congo, Democratic Republic of the", "Congo, Republic of the", "Costa Rica",
    "Cote d'Ivoire", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica",
    "Dominican Republic", "East Timor (Timor-Leste)", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea",
    "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana",
    "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland",
    "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan",
    "Kenya", "Kiribati", "Korea, North", "Korea, South", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon",
    "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia",
    "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia, Federated States of",
    "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar (Burma)", "Namibia", "Nauru", "Nepal",
    "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Macedonia (formerly Macedonia)", "Norway", "Oman",
    "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar",
    "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa",
    "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore",
    "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan",
    "Suriname", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Togo", "Tonga",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States of America", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City (Holy See)", "Venezuela",
    "Vietnam", "Yemen", "Zambia", "Zimbabwe"
]
import matplotlib.pyplot as plt
import seaborn as sns
# Assuming you have already imported necessary libraries and loaded your dataset as 'df'
# Filter user locations to include only the 195 countries
df_filtered = df[df['user_location'].isin(all_countries)]
# Visualize sentiment scores across user locations
plt.figure(figsize=(12, 6))
sns.barplot(x='user_location', y='sentiment', data=df_filtered, estimator=np.mean, ci=None)
plt.title('Average Sentiment Score Across User Locations')
plt.xlabel('User Location')
plt.ylabel('Average Sentiment Score')
plt.xticks(rotation=90, fontsize=5)  # Rotate x-axis labels by 45 degrees
plt.show()


# In[26]:


# Visualize sentiment scores across demographic groups
# For example, visualize sentiment scores across user verification status
plt.figure(figsize=(8, 6))
sns.barplot(x='user_verified', y='sentiment', data=df, ci=None)
plt.title('Average Sentiment by User Verification Status')
plt.xlabel('User Verified Status')
plt.ylabel('Average Sentiment Score')
plt.show()


# In[27]:


import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt



# Download NLTK resources
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv('covidvaccine.csv')  # Replace 'covidvaccine.csv' with the path to your dataset file

# Drop rows with missing or invalid text values
df = df.dropna(subset=['text'])
df = df[df['text'].apply(lambda x: isinstance(x, str))]

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis and add sentiment scores to the DataFrame
df['sentiment'] = df['text'].apply(lambda x: sid.polarity_scores(str(x)))

# Extract positive, negative, and neutral scores
df['positive_score'] = df['sentiment'].apply(lambda x: x['pos'])
df['negative_score'] = df['sentiment'].apply(lambda x: x['neg'])
df['neutral_score'] = df['sentiment'].apply(lambda x: x['neu'])

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

# Aggregate sentiment scores over time
daily_sentiment = df.groupby(df['date'].dt.date)[['positive_score', 'negative_score', 'neutral_score']].mean()





# Plot sentiment trends using time series plots
plt.figure(figsize=(10, 6))
daily_sentiment.plot()
plt.title('Sentiment Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.legend(['Positive', 'Negative', 'Neutral'])
plt.xticks(rotation=45)
plt.show()


# In[28]:


import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Set a modern style for plotting
plt.style.use('seaborn-darkgrid')

# Download NLTK resources
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv('covidvaccine.csv')  # Replace 'covidvaccine.csv' with the path to your dataset file

# Drop rows with missing or invalid text values
df = df.dropna(subset=['text'])
df = df[df['text'].apply(lambda x: isinstance(x, str))]

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis and add sentiment scores to the DataFrame
df['sentiment'] = df['text'].apply(lambda x: sid.polarity_scores(str(x)))

# Extract positive, negative, and neutral scores
df['positive_score'] = df['sentiment'].apply(lambda x: x['pos'])
df['negative_score'] = df['sentiment'].apply(lambda x: x['neg'])
df['neutral_score'] = df['sentiment'].apply(lambda x: x['neu'])

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

# Aggregate sentiment scores over time
daily_sentiment = df.groupby(df['date'].dt.date)[['positive_score', 'negative_score', 'neutral_score']].mean()

# Plot sentiment trends using time series plots
plt.figure(figsize=(10, 6))

# Define colors and line styles for each sentiment
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
line_styles = ['-', '--', ':']

# Plot each sentiment score with custom style
for i, (sentiment, color, line_style) in enumerate(zip(daily_sentiment.columns, colors, line_styles)):
    sns.lineplot(data=daily_sentiment[sentiment], color=color, linestyle=line_style, linewidth=2.5)

plt.title('Sentiment Trends Over Time', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Sentiment Score', fontsize=14)

plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


# In[29]:


df.columns


# In[30]:


# Assuming 'text' contains the tweet content and 'user_location' contains location information




# Derive Positive Sentiment (example using TextBlob for sentiment analysis)
from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity

df['positive_sentiment'] = df['text'].apply(get_sentiment)

# Derive Vaccination Uptake (example: frequency of vaccine-related tweets)
# Assuming 'covid_vaccine' is a keyword indicating discussion about COVID-19 vaccines
df['vaccine_related'] = df['text'].str.contains('covid_vaccine', case=False)
vaccination_uptake = df.groupby('user_location')['vaccine_related'].mean().reset_index()
vaccination_uptake.rename(columns={'vaccine_related': 'vaccination_uptake'}, inplace=True)




# Merge vaccination uptake data with the main dataframe if necessary
df = pd.merge(df, vaccination_uptake, on='user_location', how='left')






# In[31]:


from scipy.stats import pearsonr, kendalltau, pointbiserialr

# Drop rows with missing or infinite values
df.dropna(subset=['positive_sentiment', 'vaccination_uptake'], inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['positive_sentiment', 'vaccination_uptake'], inplace=True)

# Calculate correlation coefficients
pearson_corr, pearson_p_value = pearsonr(df['positive_sentiment'], df['vaccination_uptake'])
print("Pearson's correlation coefficient:", pearson_corr)

kendall_tau, kendall_p_value = kendalltau(df['positive_sentiment'], df['vaccination_uptake'])
print("Kendall's Tau correlation coefficient:", kendall_tau)

# Fill NaN values in 'user_description' column with an empty string
df['user_description'].fillna('', inplace=True)

# Example: Creating a binary variable indicating whether certain keywords are present in user descriptions
df['has_covid_mention'] = df['user_description'].str.contains('covid', case=False).astype(int)
df['has_vaccine_mention'] = df['user_description'].str.contains('vaccine', case=False).astype(int)

# Calculate point-biserial correlation coefficients for the newly created variables
point_biserial_corr_covid, _ = pointbiserialr(df['has_covid_mention'], df['positive_sentiment'])
point_biserial_corr_vaccine, _ = pointbiserialr(df['has_vaccine_mention'], df['positive_sentiment'])

print("Point-Biserial correlation coefficient for COVID mention:", point_biserial_corr_covid)
print("Point-Biserial correlation coefficient for vaccine mention:", point_biserial_corr_vaccine)

