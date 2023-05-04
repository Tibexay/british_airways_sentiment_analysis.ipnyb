# Scrapping And Analysing Customer Reviews To Uncover Findings for British Airways 

In this notebook, we will be scraping and collecting customer feedback and reviewing data from a third-party source and analysing this data to present any insights we may uncover.
We will use Python web scraping library - BeautifulSoup to scrape 2000 unique customer reviews from https://www.airlinequality.com/. 

For the analysis, we will conduct a Rule Based sentiment analysis, an approach to analyzing text without training or using machine learning models. Examples of Rule Based sentiment analysis include TextBlob, VADER, SentiWordNet.
We have chosen VADER (Valence Aware Dictionary and sEntiment Reasoner), as the python package for this analysis. Vader is used for sentiment analysis of text which has both the polarities i.e. positive/negative. VADER is used to quantify how much of positive or negative emotion the text has and also the intensity of emotion.



```python
#Install and Import webscraping packages
! pip install requests
from bs4 import BeautifulSoup
import requests


#Import data wrangling and visualization packages and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 
import seaborn as sns 

#libraries for sentiment analysis
!pip install wordcloud
from wordcloud import WordCloud
import nltk
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download([
    'stopwords',
    'punkt',
    'wordnet',
    'omw-1.4',
    'opinion_lexicon',
    'vader_lexicon'
])
plt.rcParams["figure.figsize"] = (20,10)
%matplotlib inline
```

    Requirement already satisfied: requests in c:\users\oluwa\anaconda3\lib\site-packages (2.27.1)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\oluwa\anaconda3\lib\site-packages (from requests) (1.26.9)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\oluwa\anaconda3\lib\site-packages (from requests) (2021.10.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in c:\users\oluwa\anaconda3\lib\site-packages (from requests) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\oluwa\anaconda3\lib\site-packages (from requests) (3.3)
    Requirement already satisfied: wordcloud in c:\users\oluwa\anaconda3\lib\site-packages (1.8.2.2)
    Requirement already satisfied: matplotlib in c:\users\oluwa\anaconda3\lib\site-packages (from wordcloud) (3.5.1)
    Requirement already satisfied: numpy>=1.6.1 in c:\users\oluwa\anaconda3\lib\site-packages (from wordcloud) (1.21.5)
    Requirement already satisfied: pillow in c:\users\oluwa\anaconda3\lib\site-packages (from wordcloud) (9.0.1)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\oluwa\anaconda3\lib\site-packages (from matplotlib->wordcloud) (4.25.0)
    Requirement already satisfied: cycler>=0.10 in c:\users\oluwa\anaconda3\lib\site-packages (from matplotlib->wordcloud) (0.11.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\oluwa\anaconda3\lib\site-packages (from matplotlib->wordcloud) (1.3.2)
    Requirement already satisfied: pyparsing>=2.2.1 in c:\users\oluwa\anaconda3\lib\site-packages (from matplotlib->wordcloud) (3.0.4)
    Requirement already satisfied: packaging>=20.0 in c:\users\oluwa\anaconda3\lib\site-packages (from matplotlib->wordcloud) (21.3)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\oluwa\anaconda3\lib\site-packages (from matplotlib->wordcloud) (2.8.2)
    Requirement already satisfied: six>=1.5 in c:\users\oluwa\anaconda3\lib\site-packages (from python-dateutil>=2.7->matplotlib->wordcloud) (1.16.0)
    

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\oluwa\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\oluwa\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\oluwa\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package omw-1.4 to
    [nltk_data]     C:\Users\oluwa\AppData\Roaming\nltk_data...
    [nltk_data]   Package omw-1.4 is already up-to-date!
    [nltk_data] Downloading package opinion_lexicon to
    [nltk_data]     C:\Users\oluwa\AppData\Roaming\nltk_data...
    [nltk_data]   Package opinion_lexicon is already up-to-date!
    [nltk_data] Downloading package vader_lexicon to
    [nltk_data]     C:\Users\oluwa\AppData\Roaming\nltk_data...
    [nltk_data]   Package vader_lexicon is already up-to-date!
    


```python
# Scrape data from 20 pages that contain 100 reviews each

base_url = "https://www.airlinequality.com/airline-reviews/british-airways/"
reviews = []
pages = 20
# Loop through 20 pages
for i in range(1, pages + 1):
    url = f"{base_url}/page{i}/?sortby=post_date%3ADesc&pagesize=100"
    response = requests.get(url)
    # Parse Contents
    parsed_content = BeautifulSoup(response.content, "html.parser")
    # Append the texts to reviews list
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())
print(len(reviews))
```

    2000
    


```python
# Convert reviews to a dataframe  

df = pd.DataFrame()
df["reviews"] = reviews
df["reviews"].tail()
```




    1995    ✅ Verified Review |  Flew British Airways from...
    1996    ✅ Verified Review | Flew London Gatwick to Tam...
    1997    ✅ Verified Review |  Outbound Heathrow - Frank...
    1998    ✅ Verified Review |  Dublin to San Francisco v...
    1999    ✅ Verified Review |  Mexico to Amsterdam via L...
    Name: reviews, dtype: object




```python
# Clean the dataframe of "✅ Trip Verified |" & "'✅ Verified Review |" as they are not important for the analysis

df["reviews"] = df["reviews"].astype(str).str.strip('✅ Trip Verified |')
df["reviews"] = df["reviews"].astype(str).str.strip('✅ Verified Review |')
df["reviews"] = df["reviews"].astype(str).str.strip('Not Verified |')
df["reviews"] = df["reviews"].apply(str)
```


```python
# Check Cleaned Dataframe

df["reviews"].head(20)
```




    0     Despite being a gold member, the British Airwa...
    1     Regarding the aircraft and seat: The business ...
    2     I travelled with British Airways from Sweden t...
    3     Food was lousy. Who ever is planning the Asian...
    4     Had the worst experience. The flight from Lond...
    5     he ground staff were not helpful. Felt like al...
    6     Second time BA Premium Economy in a newer airc...
    7     They changed our Flights from Brussels to Lond...
    8     At Copenhagen the most chaotic ticket counter ...
    9     Worst experience of my life trying to deal wit...
    10    Due to code sharing with Cathay Pacific I was ...
    11    LHR check in was quick at the First Wing and q...
    12    I wouldn't recommend British Airways at all. I...
    13    Absolutely horrible experience. I booked a tic...
    14    This is the worst airline. Not one thing went ...
    15    I will never fly British Airways again. To sta...
    16    Worst aircraft I have ever flown. The seats we...
    17    I enjoyed my flight. The boarding was swift an...
    18    Why do you make it so hard? After a so so loun...
    19    After several delays and canceled flights, we ...
    Name: reviews, dtype: object




```python
# Instantiate VADER's sentiment analyzer
sentiment = SentimentIntensityAnalyzer()
```


```python
# Generate Polarity scores for the reviews
df['compound'] = [sentiment.polarity_scores(review)['compound'] for review in df['reviews']]
df['neg'] = [sentiment.polarity_scores(review)['neg'] for review in df['reviews']]
df['neu'] = [sentiment.polarity_scores(review)['neu'] for review in df['reviews']]
df['pos'] = [sentiment.polarity_scores(review)['pos'] for review in df['reviews']]
```


```python
# Inspect dataframe for results

df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reviews</th>
      <th>compound</th>
      <th>neg</th>
      <th>neu</th>
      <th>pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Despite being a gold member, the British Airwa...</td>
      <td>-0.8680</td>
      <td>0.161</td>
      <td>0.781</td>
      <td>0.058</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Regarding the aircraft and seat: The business ...</td>
      <td>-0.4541</td>
      <td>0.101</td>
      <td>0.813</td>
      <td>0.086</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I travelled with British Airways from Sweden t...</td>
      <td>-0.9455</td>
      <td>0.069</td>
      <td>0.903</td>
      <td>0.027</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Food was lousy. Who ever is planning the Asian...</td>
      <td>-0.7476</td>
      <td>0.110</td>
      <td>0.842</td>
      <td>0.049</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Had the worst experience. The flight from Lond...</td>
      <td>-0.8604</td>
      <td>0.110</td>
      <td>0.866</td>
      <td>0.023</td>
    </tr>
    <tr>
      <th>5</th>
      <td>he ground staff were not helpful. Felt like al...</td>
      <td>-0.8537</td>
      <td>0.166</td>
      <td>0.834</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Second time BA Premium Economy in a newer airc...</td>
      <td>0.9714</td>
      <td>0.052</td>
      <td>0.677</td>
      <td>0.270</td>
    </tr>
    <tr>
      <th>7</th>
      <td>They changed our Flights from Brussels to Lond...</td>
      <td>-0.8055</td>
      <td>0.096</td>
      <td>0.866</td>
      <td>0.038</td>
    </tr>
    <tr>
      <th>8</th>
      <td>At Copenhagen the most chaotic ticket counter ...</td>
      <td>0.1015</td>
      <td>0.102</td>
      <td>0.756</td>
      <td>0.142</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Worst experience of my life trying to deal wit...</td>
      <td>-0.9600</td>
      <td>0.152</td>
      <td>0.798</td>
      <td>0.049</td>
    </tr>
  </tbody>
</table>
</div>



_The compound score is the sum of positive, negative & neutral scores which is then normalized between -1(most extreme negative) and +1 (most extreme positive)._

_The pos, neu, and neg scores are ratios for proportions of text that fall in each category (so these should all add up to be 1, or close to it with float operation). These are the most useful metrics if you want to analyze the context & presentation of how sentiment is conveyed or embedded in rhetoric for a given sentence._


```python
# Conduct Descriptive analysis of the result
df[['compound', 'neg', 'neu', 'pos']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound</th>
      <th>neg</th>
      <th>neu</th>
      <th>pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.086998</td>
      <td>0.083536</td>
      <td>0.811913</td>
      <td>0.104560</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.785339</td>
      <td>0.055097</td>
      <td>0.072763</td>
      <td>0.079436</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.997400</td>
      <td>0.000000</td>
      <td>0.436000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.783100</td>
      <td>0.044750</td>
      <td>0.770000</td>
      <td>0.048000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.250800</td>
      <td>0.076000</td>
      <td>0.818000</td>
      <td>0.087000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.909950</td>
      <td>0.114000</td>
      <td>0.863000</td>
      <td>0.145000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.999200</td>
      <td>0.348000</td>
      <td>1.000000</td>
      <td>0.490000</td>
    </tr>
  </tbody>
</table>
</div>



_From the results, we can see with median compound score is 0.30 which means that 50% of the reviews have a compound score of more than  0.30, which suggests a positive sentiment._


```python
# Visualiziling compound score results

sns.histplot(df['compound'])
plt.title("Distribution of Compound Scores")
plt.savefig("Compound.png");
```


    
![png](british_airways_sentiment_analysis_files/british_airways_sentiment_analysis_13_0.png)
    


_The peaks at the extremes suggest that, the are many reviews that are either extremely positive or extremely negative, and there are relatively few moderate and nuetral reviews_


```python
# Classify sentiments as positive, negative and neutral

def compute_sentiment(df):
    if df['compound'] >= 0.05:
        return "Positive"
    elif df['compound'] <= - 0.05:
        return "Negative"
    else:
        return "Neutral"
    
df["sentiment"] = df.apply(compute_sentiment, axis = 1)

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reviews</th>
      <th>compound</th>
      <th>neg</th>
      <th>neu</th>
      <th>pos</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Despite being a gold member, the British Airwa...</td>
      <td>-0.8680</td>
      <td>0.161</td>
      <td>0.781</td>
      <td>0.058</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Regarding the aircraft and seat: The business ...</td>
      <td>-0.4541</td>
      <td>0.101</td>
      <td>0.813</td>
      <td>0.086</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I travelled with British Airways from Sweden t...</td>
      <td>-0.9455</td>
      <td>0.069</td>
      <td>0.903</td>
      <td>0.027</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Food was lousy. Who ever is planning the Asian...</td>
      <td>-0.7476</td>
      <td>0.110</td>
      <td>0.842</td>
      <td>0.049</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Had the worst experience. The flight from Lond...</td>
      <td>-0.8604</td>
      <td>0.110</td>
      <td>0.866</td>
      <td>0.023</td>
      <td>Negative</td>
    </tr>
  </tbody>
</table>
</div>



_the chart shows the proportions 


```python
df["sentiment"].value_counts(normalize=True)*100
```




    Positive    53.15
    Negative    45.30
    Neutral      1.55
    Name: sentiment, dtype: float64




```python
sns.countplot(x=df["sentiment"])
plt.title("Proportion of Sentiments in the Reviews")
plt.rcParams["figure.figsize"] = (20,10)
plt.savefig("proportion.png");
```


    
![png](british_airways_sentiment_analysis_files/british_airways_sentiment_analysis_18_0.png)
    


_Bar Chart shows a clearly that there are more positive reviews than there are negative ones_




### Creating word clouds for the positive and negative reviews for extract insights 



```python
# Def Stop words
stop_words = nltk.corpus.stopwords.words('english')

#Create a function to preprocess the data for wordcloud visualization
def preprocess_text(text):
    tokenized_document = nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9\']+').tokenize(text) # Tokenize
    cleaned_tokens = [word.lower() for word in tokenized_document if word.lower() not in stop_words] # remove and turn to lower case
    stemmed_text = [nltk.stem.PorterStemmer().stem(word) for word in cleaned_tokens] # stemming
    return stemmed_text

df["processed_reviews"] = df["reviews"].apply(preprocess_text)
```


```python
# Subset data
positive_reviews = df.loc[(df["sentiment"]=="Positive"),:]
negative_reviews = df.loc[(df["sentiment"]=="Negative"),:]
```


```python
pos_tokens = (word for df in positive_reviews["processed_reviews"] for word in df)
wordcloud = WordCloud(background_color='white').generate_from_text(' '.join(pos_tokens))

plt.figure(figsize=(12,14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
plt.savefig("positivewc.png")
```


    
![png](british_airways_sentiment_analysis_files/british_airways_sentiment_analysis_23_0.png)
    



    <Figure size 1440x720 with 0 Axes>



```python
neg_tokens = (word for df in negative_reviews["processed_reviews"] for word in df)

wordcloud1 = WordCloud(background_color='white').generate_from_text(' '.join(neg_tokens))


plt.figure(figsize=(12,14))
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis('off')
plt.title('Negative wordcloud')
plt.show()
```


    
![png](british_airways_sentiment_analysis_files/british_airways_sentiment_analysis_24_0.png)
    



```python
pip install nbconvert
```

    Requirement already satisfied: nbconvert in c:\users\oluwa\anaconda3\lib\site-packages (6.4.4)
    Requirement already satisfied: jupyter-core in c:\users\oluwa\anaconda3\lib\site-packages (from nbconvert) (4.9.2)
    Requirement already satisfied: pandocfilters>=1.4.1 in c:\users\oluwa\anaconda3\lib\site-packages (from nbconvert) (1.5.0)
    Requirement already satisfied: testpath in c:\users\oluwa\anaconda3\lib\site-packages (from nbconvert) (0.5.0)
    Requirement already satisfied: entrypoints>=0.2.2 in c:\users\oluwa\anaconda3\lib\site-packages (from nbconvert) (0.4)
    Requirement already satisfied: beautifulsoup4 in c:\users\oluwa\anaconda3\lib\site-packages (from nbconvert) (4.11.1)
    Requirement already satisfied: nbformat>=4.4 in c:\users\oluwa\anaconda3\lib\site-packages (from nbconvert) (5.3.0)
    Requirement already satisfied: pygments>=2.4.1 in c:\users\oluwa\anaconda3\lib\site-packages (from nbconvert) (2.11.2)
    Requirement already satisfied: nbclient<0.6.0,>=0.5.0 in c:\users\oluwa\anaconda3\lib\site-packages (from nbconvert) (0.5.13)
    Requirement already satisfied: mistune<2,>=0.8.1 in c:\users\oluwa\anaconda3\lib\site-packages (from nbconvert) (0.8.4)
    Requirement already satisfied: jupyterlab-pygments in c:\users\oluwa\anaconda3\lib\site-packages (from nbconvert) (0.1.2)
    Requirement already satisfied: defusedxml in c:\users\oluwa\anaconda3\lib\site-packages (from nbconvert) (0.7.1)
    Requirement already satisfied: jinja2>=2.4 in c:\users\oluwa\anaconda3\lib\site-packages (from nbconvert) (2.11.3)
    Requirement already satisfied: traitlets>=5.0 in c:\users\oluwa\anaconda3\lib\site-packages (from nbconvert) (5.1.1)
    Requirement already satisfied: bleach in c:\users\oluwa\anaconda3\lib\site-packages (from nbconvert) (4.1.0)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\users\oluwa\anaconda3\lib\site-packages (from jinja2>=2.4->nbconvert) (2.0.1)
    Requirement already satisfied: jupyter-client>=6.1.5 in c:\users\oluwa\anaconda3\lib\site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert) (6.1.12)
    Requirement already satisfied: nest-asyncio in c:\users\oluwa\anaconda3\lib\site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert) (1.5.5)
    Requirement already satisfied: tornado>=4.1 in c:\users\oluwa\anaconda3\lib\site-packages (from jupyter-client>=6.1.5->nbclient<0.6.0,>=0.5.0->nbconvert) (6.1)
    Requirement already satisfied: python-dateutil>=2.1 in c:\users\oluwa\anaconda3\lib\site-packages (from jupyter-client>=6.1.5->nbclient<0.6.0,>=0.5.0->nbconvert) (2.8.2)
    Requirement already satisfied: pyzmq>=13 in c:\users\oluwa\anaconda3\lib\site-packages (from jupyter-client>=6.1.5->nbclient<0.6.0,>=0.5.0->nbconvert) (22.3.0)
    Requirement already satisfied: pywin32>=1.0 in c:\users\oluwa\anaconda3\lib\site-packages (from jupyter-core->nbconvert) (302)
    Requirement already satisfied: fastjsonschema in c:\users\oluwa\anaconda3\lib\site-packages (from nbformat>=4.4->nbconvert) (2.15.1)
    Requirement already satisfied: jsonschema>=2.6 in c:\users\oluwa\anaconda3\lib\site-packages (from nbformat>=4.4->nbconvert) (4.4.0)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in c:\users\oluwa\anaconda3\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4->nbconvert) (0.18.0)
    Requirement already satisfied: attrs>=17.4.0 in c:\users\oluwa\anaconda3\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4->nbconvert) (21.4.0)
    Requirement already satisfied: six>=1.5 in c:\users\oluwa\anaconda3\lib\site-packages (from python-dateutil>=2.1->jupyter-client>=6.1.5->nbclient<0.6.0,>=0.5.0->nbconvert) (1.16.0)
    Requirement already satisfied: soupsieve>1.2 in c:\users\oluwa\anaconda3\lib\site-packages (from beautifulsoup4->nbconvert) (2.3.1)
    Requirement already satisfied: webencodings in c:\users\oluwa\anaconda3\lib\site-packages (from bleach->nbconvert) (0.5.1)
    Requirement already satisfied: packaging in c:\users\oluwa\anaconda3\lib\site-packages (from bleach->nbconvert) (21.3)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in c:\users\oluwa\anaconda3\lib\site-packages (from packaging->bleach->nbconvert) (3.0.4)
    Note: you may need to restart the kernel to use updated packages.
    


```python
jupyter nbconvert --to markdown british_airways_sentiment_analysis.ipynb
```


      Input In [19]
        jupyter nbconvert --to markdown british_airways_sentiment_analysis.ipynb
                ^
    SyntaxError: invalid syntax
    



```python

```
