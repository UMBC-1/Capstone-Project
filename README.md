# 606 - Capstone Project 

# Project title 

# Enhancing Information Integrity A Machine Learning and NLP Approach to Fake News Detection


## Introduction
  
The modern technology era of internet and social media platfomrs the dataflow is increased and growing day-by-day. Even though there are lot of changes and challenges have to be faced because of the incorrect facts and figures. The fake news will spread rapidly and attract people mindset which will further change the readers thoughts and future discussions. To eradicate this, using NLP techniques and ML techniques wil be a great solution and innovate approach to solve this issue. To face this we require the combinational approach which helps to get highest accuracy with accurate predictional output.

ML models have great ability to analyze the data and make the predictions with accurate percentage output. This huge amount of dataflow is controlled by using different enhanced tools and softwares. To find the news is fake and true we need to understand the human language of words where the data should align in this dimensions which will increase the accuracy rate of the NLP algorithms. 

The aim of this project is to establish a connection between the data and human which helps to understand human configurations to test the data is fake ot true. The dataset used to train the data is going to teach the commonly used characters and the informations whcih is missed. The ieratetive approach is refined and tested on the data which able to create a new model and gives accuarte results that creates new trustworthy source of information by eradicating the fake news. The goal is to increase the accurate information in this digital media. To combat the fake news we need to use this innovative approachable technologies more and handle in efficient manner 

﻿Through this undertaking, we no longer handiest aim to broaden a practical option to the hassle of fake information however additionally to elevate recognition about the importance of essential questioning and media literacy within the digital age. By empowering customers to make informed judgments approximately the records they come across, we will together work in the direction of a more truthful and transparent information surroundings. With this creation, we lay the basis for our exploration of faux news detection via machine studying and NLP. As we delve deeper into the technical elements of our method and percentage our findings and insights, we invite fellow researchers, developers, and enthusiasts to join us in our quest to uphold the integrity of facts in the virtual generation.

## Literature Review

﻿The proliferation of fake information in present day virtual panorama has underscored the important want for powerful detection methods. Various studies have been performed to deal with this pressing trouble, employing various procedures and strategies. This literature assessment offers a top level view of key studies inside the field of faux news detection:

1. "Fake News Detection on Social Media: A Data Mining Perspective" with the aid of Shu et al. (2017):
Shu et al. Proposed a statistics mining framework for detecting faux information on social media platforms. Their approach incorporated features along with linguistic traits, person engagement, and propagation styles to differentiate between faux and true news. The look at applied Logistic Regression and Natural Language Processing (NLP) strategies, specially Term Frequency-Inverse Document Frequency (TF-IDF), to research and classify information content.

2. ﻿"Fake News Detection: A Deep Learning Approach" by means of Patel et al. (2018):
Patel et al. Added a deep mastering-primarily based technique for faux information detection, leveraging Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. Their research showcased the efficacy of deep mastering fashions in discerning faux information from legitimate resources. The study employed Multi-Layer Perceptron and NLP strategies, focusing specifically on stance detection to discover misleading statistics.

3. ﻿"Detection of Fake News in Social Media Networks" by way of Wang et al. (2018):
Wang et al. Conducted a complete survey of fake information detection techniques deployed in social media networks. Their take a look at encompassed content-primarily based, person-based, and propagation-primarily based processes, analyzing the strengths and barriers of each method. Classification algorithms, Python scripts, and statistical methods were employed to assess and validate the effectiveness of the proposed techniques.

4. ﻿"A Survey of Fake News Detection Methods" through Sharma et al. (2019):
Sharma et al. Supplied an in depth survey of faux information detection methodologies, categorizing them into linguistic, community-based, and hybrid methods. The study elucidated the demanding situations related to each technique and identified potential avenues for future studies and improvement. The survey encompassed a extensive variety of solutions and mitigation strategies hired in combating the unfold of incorrect information.

5. ﻿"Fake News Detection Using Natural Language Processing" by means of Jin et al. (2020):
Jin et al. Proposed a faux information detection framework primarily based on Natural Language Processing (NLP) techniques. Their methodology concerned extracting linguistic features from news articles and employing device studying algorithms, particularly Logistic Regression with Count Vectorizer, to classify news content material as both fake or real.

These studies collectively contribute to advancing the field of fake news detection, offering insights into diverse methodologies and approaches aimed at mitigating the harmful effects of misinformation.

# Dataset Overview

﻿The dataset contains a set of news articles available on Kaggle, on hand via the subsequent link: [Fake News Detection Dataset](https://www.Kaggle.Com/code/therealsampat/fake-news-detection/enter). It consists of  predominant documents:

1. Fake.csv:

- File Size: 62.79MB
* Number of Rows: 23,481
+ Number of Columns: 4

2. True.csv:

- File Size: 53.58MB
* Number of Rows: 21,417
+ Number of Columns: 4

﻿These documents contain a based dataset of information articles, labeled into fake and proper news. The information offers precious resources for researchers and practitioners interested in analyzing and growing techniques for fake information detection.

# Workflow of the Project
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/a9d22fd4-9977-40c7-a853-02475ee9a13e)

1. Data Acquisition and Understanding

﻿Dataset Selection: Identify the statistics source applicable in your challenge's dreams. This statistics can come from various resources like internal databases, internet scraping, APIs, or public datasets.
Data Understanding: Explore and examine the preliminary information to understand its characteristics. This includes facts sorts, presence of missing values, outliers, and capacity inconsistencies.

2. Data Cleaning and Preprocessing

﻿Missing Value Handling: Address lacking information points the usage of strategies like mean/median imputation, deletion, or wearing ahead/backward values based totally on context.
Outlier Treatment: Identify and deal with outliers that could skew your analysis. This may also involve capping outliers to a specific variety, winsorizing, or elimination depending at the scenario.
Data Cleaning: Address inconsistencies like typos, formatting mistakes, or unique characters to make certain statistics great.
Normalization/Scaling: Normalize or scale features (columns) on your information to a common range to save you precise functions from dominating the version education system.
Dimensionality Reduction (Optional): If you have got a high number of features, consider dimensionality reduction techniques like Principal Component Analysis (PCA) to reduce the function area whilst maintaining most information.

3. Feature Engineering and NLP Tasks

﻿Feature Engineering: Create new functions from current ones that might be more informative on your models. This can contain feature creation, interaction phrases, or binning categorical features.
Natural Language Processing (NLP Tasks): If your statistics is textual content-primarily based, carry out NLP obligations like tokenization (splitting text into phrases), stemming (lowering words to their root form), or lemmatization (converting phrases to their dictionary form) to prepare the text information for modeling.

4. Feature Extraction

﻿Select a subset of relevant functions from your statistics so as to be utilized by the gadget mastering models for education. Feature selection strategies like correlation evaluation, chi-rectangular checks, or feature importance ratings can help identify the maximum informative functions.

5. Model Training and Selection

﻿Data Splitting: Divide your cleaned and preprocessed statistics into  sets: a education set (used to train the fashions) and a testing set (used to evaluate the fashions' overall performance on unseen facts).
Algorithm Selection: Choose one or more gadget learning algorithms appropriate for your problem type. The flowchart offers examples like Logistic Regression (type issues), Support Vector Machine (class or regression), Decision Tree (class or regression), and Random Forest (classification or regression).
Model Training: Train each selected version at the training set. The version learns patterns and relationships in the records to make predictions.

6. Model Evaluation

﻿Evaluation Metrics: Evaluate the overall performance of the skilled fashions on the checking out set the usage of metrics like accuracy, precision, consider, F1 score, or different applicable metrics depending to your trouble (class vs. Regression). This facilitates you apprehend how nicely the version generalizes to unseen information.
Model Selection: Based on the evaluation metrics, pick the model with the nice overall performance in your precise desires.

7. Model Deployment

﻿Web Application Development: Develop an internet application to installation your preferred version. This permits users to interact with the version and reap predictions based on new input information.
 
# New Developments

In the realm of text analysis, the integration of natural language processing (NLP) techniques during pre-processing is crucial for extracting meaningful insights from textual data. Techniques such as tokenization, stop word removal, and lemmatization or stemming are commonly employed to prepare the text for analysis. However, to delve deeper into the semantic relationships between words, leveraging advanced methods like Word2Vec embedding proves invaluable.

Word2Vec enables the transformation of words into dense vectors, capturing semantic similarities and nuances in meaning. By representing words in a continuous vector space, Word2Vec facilitates tasks such as word similarity and analogy, enhancing the depth of analysis. Moreover, to address the computational challenges posed by large datasets, parallel processing techniques can be employed. By distributing the workload across multiple processors or cores, parallel processing significantly improves efficiency and reduces processing time.

In addition to leveraging sophisticated techniques, providing a user-friendly interface is essential for facilitating access to the analysis tools. Streamlit, a powerful Python library, offers a convenient solution for creating interactive web applications with minimal effort. By integrating Streamlit into the analysis pipeline, users can effortlessly interact with the data, visualize results, and explore insights in real-time.

In summary, the synergy between NLP techniques, Word2Vec embedding, parallel processing, and Streamlit interface not only enhances the efficiency of text analysis but also empowers users to uncover deeper insights from textual data with ease.

# Data Preprocessing

In the process of preparing our data for analysis, we began by merging two distinct datasets: one comprising real data and the other containing fake data. By consolidating these datasets, we aimed to create a comprehensive corpus that encompasses a diverse range of textual content.

Subsequently, we introduced a target variable to each dataset to facilitate classification tasks. Specifically, we assigned a value of 1 to instances originating from the real data, signifying their authenticity, while instances sourced from the fake data were labeled with a value of 0.

To ensure the integrity of our dataset and maintain data quality, we conducted thorough data cleaning procedures. This included identifying and addressing any null or missing values present within the combined dataset. By meticulously removing or imputing missing values, we fortified the robustness of our dataset, thereby enhancing the reliability and accuracy of subsequent analyses.

# Step-by-step Code Analysis
## Importing Packages

```<python>
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
```

Importing Essential Libraries for Data Science Tasks

This code block imports several libraries that are crucial for data science projects:
1. pandas (pd): This library is a workhorse for data manipulation and analysis. It provides powerful data structures like DataFrames and Series, along with functions for data cleaning, transformation, and exploration.
2. numpy (np): This library offers efficient tools for numerical computations. It's essential for performing array-based operations, linear algebra calculations, and other mathematical tasks commonly encountered in data science.
3. seaborn (sns): This library builds on top of matplotlib to create high-level visualizations. It simplifies the creation of informative and visually appealing charts and graphs, allowing you to effectively communicate insights from your data analysis.
4. nltk: The Natural Language Toolkit (nltk) provides functionalities for working with text data. It offers tools for tokenization (splitting text into words), stemming (reducing words to their root form), sentiment analysis, and other NLP tasks.
5. stopwords from nltk.corpus: This specifically imports the English stop words list from the NLTK corpus. Stop words are common words like "the," "a," or "an" that carry little meaning on their own. By removing them during text analysis, you can focus on the more informative content within your text data.
6. plotly.express (px): This library provides a user-friendly way to create interactive visualizations in Python. It offers a wide range of chart types like scatter plots, bar charts, line charts, and more, with the ability to add interactivity for users to explore the data dynamically.

By importing these libraries, you equip your data science project with the necessary tools for data handling, manipulation, visualization, and potentially working with textual data.

## Dataset Importing from Drive

```<python>
from google.colab import drive
drive.mount('/content/drive')

col = ["title", "text", "subject", "date"]
fake_data = pd.read_csv('/content/capstone/Fake.csv', header=None, names=col, skiprows=1)
real_data = pd.read_csv('/content/capstone/True.csv', header=None, names=col, skiprows=1)
```
## Display the Datasets

```<python>
fake_data.head()
```
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/b3cf8e91-3b1b-411c-979c-9e19a3df6b32)
```<python>
fake_data.info()
```
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/9091b7cc-bfbf-43f7-b644-5e89da59d476)
```<python>
real_data.head()
```
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/b0b46150-7afc-4718-a299-23ce8016a209)
```<python>
real_data.info()
```
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/0adc5446-080a-4abe-8d6b-28151a759a77)
```<python>
real_data['category'] = 1
fake_data['category'] = 0
```
This code (real_data=1, fake_data=0) assigns labels (1=real, 0=fake) to categories in separate datasets (real_data and fake_data). This converts textual categories into numerical values for machine learning models to process the data.
```<python>
df = pd.DataFrame(pd.concat([fake_data,real_data]))
```
This line of code merges two DataFrames, fake_data and real_data, into a new DataFrame named df, likely stacking them vertically to combine their rows.
```<python>
df.head()
```
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/f710876f-6119-4986-8234-f3d55a2cddf5)
```<python>
df.isnull().sum()
```
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/234b182c-c285-4c2b-9567-15e2c9d8a58a)

This concludes there are no null values.

# Exploratory Data Analysis
## Step-by-step Code Analysis
```<python>
from matplotlib import pyplot as plt
df['subject'].value_counts().plot(kind='barh', color='red')
plt.xlabel('Subject')
plt.ylabel('Frequency')
plt.title('Frequency of subjects')
plt.show()
```
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/90d2aace-64a9-49f9-a918-02cc50d8c603)

This bar graph depicts subject frequency, likely in a news article dataset. The x-axis lists subjects like "US News" and "Politics", with the y-axis showing their frequency (possibly number of articles). Red and white bars represent two categories (unclear from missing legend), with "US News" and "Politics" being the most frequent subjects overall.
```<python>
plt.figure(figsize=(8, 6))
df['category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Pie Chart of Categories')
plt.ylabel('')
plt.show()
```
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/3826b1dd-339e-4bbe-948a-99f355ed83c1)

This pie chart likely depicts the distribution of real and fake data in your dataset. With blue labeled as 52.3% and orange at 47.7%, the data seems fairly balanced between real and fake categories. 
```<python>
# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'],errors='coerce')

timeline_data = df.groupby([df['date'].dt.to_period('M'), 'category']).size().unstack().fillna(0)
```
This code snippet performs several key tasks to create a visualization of fake and real news articles over time:

1. Converting 'date' to Datetime:

- df['date'] = pd.to_datetime(df['date'],errors='coerce'): This line converts the 'date' column in your DataFrame df into a proper datetime format using the pandas.to_datetime function. The errors='coerce' argument ensures that any non-convertible dates are set to NaT (Not a Time) instead of raising errors.

2. Creating the Timeline Data:

- timeline_data = df.groupby([df['date'].dt.to_period('M'), 'category']).size().unstack().fillna(0): This part creates a new DataFrame named timeline_data that summarizes the data by month and category:
  - df.groupby([df['date'].dt.to_period('M'), 'category']): This groups the data in df by two factors: the month ('M') extracted from the datetime column         
  (df['date'].dt.to_period('M')) and the category ('category').
  - size(): This calculates the count of entries within each group, essentially giving you the number of real/fake news articles for each month.
  - unstack(): This transforms the grouped data into a DataFrame with months as rows and categories ('category') as columns. The resulting DataFrame shows the         monthly counts of real and fake news articles.
  - fillna(0): This fills any missing values (months with no articles) in the DataFrame with 0, ensuring a complete timeline for visualization.

Importance of Timeline Conversion:

Converting the date to a timeline format is crucial for creating the desired visualization:

- Time-based Analysis: By converting the date to a monthly period, you can analyze and visualize how the number of real and fake news articles changes over time. Looking at raw dates wouldn't provide a clear picture of these trends.

* Stacked Area Chart: The code uses timeline_data.plot(kind='area',stacked=True) to create a stacked area chart. This chart effectively portrays the volume of real and fake news articles (represented by areas) throughout the months. Without the timeline conversion, such an informative visualization wouldn't be possible.

In essence, converting the date to a timeline allows you to explore and visualize temporal patterns in your data, revealing trends and potential relationships between real/fake news distribution and time.
```<python>
plt.figure(figsize=(15, 6))
timeline_data.plot(kind='area',stacked=True, color=['lightpink','lightblue'])
plt.title('Timeline of Fake and Real News Articles')
plt.xlabel('Month')
plt.ylabel('Number of Articles')
plt.legend(title='Fake or Real')
plt.show()
```
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/3b5c3111-4e52-4824-b83d-e422452ea081)

This time series plot visualizes the distribution of fake (light pink) and real (light blue) news articles over time (months). The stacked area chart shows the cumulative number of articles in each category, with color intensity potentially indicating higher volumes of fake news articles compared to real news articles throughout the displayed timeframe.
```<python>
df['color'] = df['category'].map({0: 'salmon ', 1: 'lightblue'})

df['text_length'] = df['text'].apply(lambda x: len(x))

fig_histogram = px.histogram(df, x='text_length', color='category',
                             color_discrete_map={0: 'lightsalmon', 1: 'lightblue'},
                             marginal='box', # Displays a box plot for additional insight
                             title='Distribution of Article Lengths by Category')
fig_histogram.show()
```
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/80763949-3eb1-4156-b5c6-00169bd51335)

## Text Processing
```<python>
fake = " ".join(article for article in fake_data["text"])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove digits
    return text

fake_cleaned = preprocess_text(fake)

fake_cleaned[:300]
```
This code snippet prepares text data from your "fake_data" for further analysis, likely related to creating a word cloud of fake news articles. Here's a breakdown:

1. Combining Text:
- fake = " ".join(article for article in fake_data["text"]): This line merges all the text content from the "text" column in your "fake_data" DataFrame into a single string named "fake". This combines the text from all fake news articles.

2. Preprocessing Function:
- def preprocess_text(text):: This defines a function named preprocess_text that takes a text string as input.
Text Cleaning Steps:
- text = text.lower(): This line converts all characters in the text to lowercase for consistency.
text = re.sub(r'[^\w\s]', '', text): This uses regular expressions (re.sub) to remove all characters except alphanumeric characters (\w) and whitespace (\s). This eliminates punctuation marks from the text.
- text = re.sub(r'\d+', '', text): Another regular expression removes digits (\d+) from the text, focusing on the words themselves.

3. Cleaning Applied:
- fake_cleaned = preprocess_text(fake): This line applies the preprocess_text function to the combined text string "fake", cleaning it by removing punctuation and digits.

5. Output:
- fake_cleaned[:300]: This line displays the first 300 characters of the cleaned text "fake_cleaned", allowing you to see a sample of the processed text.

Overall, this code prepares the text data from fake news articles for further analysis by converting it to lowercase, removing punctuation and digits, and potentially feeding it into a word cloud creation process to visualize frequently used words within the fake news content.

Output: 
'donald trump just couldn t wish all americans a happy new year and leave it at that instead he had to give a shout out to his enemies haters and  the very dishonest fake news media  the former reality show star had just one job to do and he couldn t do it as our country rapidly grows stronger and sm'
```<python>
wordcloud_fake = WordCloud(stopwords=STOPWORDS, background_color="white", max_words=500, width=800, height=400).generate(fake_cleaned)

fig = go.Figure(go.Image(z=wordcloud_fake))
fig.update_layout(title_text='Word Cloud for Fake News Articles', title_x=0.5)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()
```
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/7de985a9-8f9b-4d52-9c46-431966a89303)
```<python>
real = " ".join(article for article in real_data["text"])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove digits
    return text

real_cleaned = preprocess_text(real)

real_cleaned[:300]
```
Output: 
'washington reuters  the head of a conservative republican faction in the us congress who voted this month for a huge expansion of the national debt to pay for tax cuts called himself a fiscal conservative on sunday and urged budget restraint in  in keeping with a sharp pivot under way among republic'
```<python>
wordcloud_real = WordCloud(stopwords=STOPWORDS, background_color="white", max_words=500, width=800, height=400).generate(real_cleaned)

fig = go.Figure(go.Image(z=wordcloud_real))
fig.update_layout(title_text='Word Cloud for Real News Articles', title_x=0.5)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()
```
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/a24ba6aa-1f17-43cb-9bbd-57a3285dcf75)
```<python>
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

# Plotting word count distribution for real text
real_word_count = df[df['category'] == 1]['text'].str.split().apply(len)
ax1.hist(real_word_count, color='blue', bins=20)
ax1.set_title('Real Text Word Count Distribution')

# Plotting word count distribution for fake text
fake_word_count = df[df['category'] == 0]['text'].str.split().apply(len)
ax2.hist(fake_word_count, color='orange', bins=20)
ax2.set_title('Fake Text Word Count Distribution')

# Adding labels and titles
ax1.set_xlabel('Word Count')
ax1.set_ylabel('Frequency')
ax2.set_xlabel('Word Count')
ax2.set_ylabel('Frequency')
fig.suptitle('Word Count Distribution for Real and Fake Text')

# Show the plot
plt.show()
```
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/a950e61b-02c6-4631-b4dc-c3bb4313313c)
```<python>
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

# Plotting word count distribution for real text
real_char_count = df[df['category'] == 1]['text'].str.len()
ax1.hist(real_char_count, color='blue', bins=20)
ax1.set_title('Real Text character Count Distribution')

# Plotting word count distribution for fake text
fake_word_count = df[df['category'] == 0]['text'].str.len()
ax2.hist(fake_word_count, color='orange', bins=20)
ax2.set_title('Fake Text character Count Distribution')

# Adding labels and titles
ax1.set_xlabel('Count')
ax1.set_ylabel('Frequency')
ax2.set_xlabel('Count')
ax2.set_ylabel('Frequency')
fig.suptitle('Character Count Distribution for Real and Fake Text')

# Show the plot
plt.show()
```
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/371082d6-6eb5-4c6a-bf7c-e96e157f8f4e)

# Model Development
## N GRAM Analysis

N-grams are a fundamental concept in the realm of text analysis, defined as contiguous sequences of n items extracted from a given sample of text or speech. These items can vary depending on the application and context, encompassing letters, words, or even base pairs in the case of genomic analysis. The utility of N-grams extends across various domains, with applications ranging from natural language processing to bioinformatics. In text analysis, N-grams serve as building blocks for tasks such as language modeling, sentiment analysis, and predictive text input. By capturing the sequential relationships between elements within a text, N-grams provide valuable insights into linguistic patterns and structures.

For instance, in the context of language modeling, N-grams are utilized to estimate the probability of encountering a particular sequence of words within a given corpus. This probabilistic approach forms the basis for applications such as text prediction and autocorrection in mobile keyboards and search engines. Moreover, N-grams play a crucial role in the detection of plagiarism and authorship attribution by identifying similarities in textual patterns across documents. By analyzing the frequency and distribution of N-grams, researchers can assess the uniqueness and authenticity of written content, thereby aiding in the preservation of academic integrity and intellectual property rights. In summary, N-grams serve as versatile tools in text analysis, enabling the extraction of meaningful insights from textual data across diverse fields and applications. Through their ability to capture sequential relationships and patterns, N-grams empower researchers and practitioners to unravel the complexities of language and communication in both spoken and written forms.

### Step-by-step Analysis
```<python>
from nltk.util import ngrams
from collections import Counter

def get_ngrams(corpus, ngram_range, top_n):
    # Tokenize the corpus
    tokenized_corpus = [word for sentence in corpus for word in sentence.split()]
    # Generate n-grams
    n_grams = ngrams(tokenized_corpus, ngram_range)
    # Count the frequency of each n-gram
    ngram_freq = Counter(n_grams)

    # Get top n-grams
    top_ngrams = ngram_freq.most_common(top_n)

    return top_ngrams

# Get top 5 bi-grams from fake news titles
fake_news_bigrams = get_ngrams(fake_data['title'],2, 5)

# Get top 5 bi-grams from true news titles
true_news_bigrams = get_ngrams(real_data['title'],2, 5)

# Create DataFrame
bigrams_df = pd.DataFrame({
    'Fake News Bi-Grams': [f"{bigram[0]} ({bigram[1]})" for bigram in fake_news_bigrams],
    'True News Bi-Grams': [f"{bigram[0]} ({bigram[1]})" for bigram in true_news_bigrams]
})

bigrams_df
```
This code snippet analyzes bigrams (2-word phrases) in fake and real news article titles, identifying the most frequent ones. Here's a breakdown of what each part does:

1. Importing Libraries:

- from nltk.util import ngrams: This imports the ngrams function from the nltk.util module, which helps generate sequences of N words (N-grams) from a text corpus.
* from collections import Counter: This imports the Counter class from the collections module, which is used to create a dictionary-like object that keeps track of how often each element appears in an iterable.

2. get_ngrams Function:

- def get_ngrams(corpus, ngram_range, top_n):: This defines a function named get_ngrams that takes three arguments:

  - corpus: This is the text data you want to analyze, likely a list of sentences or titles in your case.
  - ngram_range: This is a tuple specifying the range of N-gram lengths to consider (e.g., (2, 2) for bigrams).
  - top_n: This is an integer representing the number of most frequent N-grams to return.
- tokenized_corpus = [word for sentence in corpus for word in sentence.split()]: This line iterates through the corpus, tokenizes each sentence (splitting it into words), and creates a flat list named tokenized_corpus that contains all the individual words.

- n_grams = ngrams(tokenized_corpus, ngram_range): This line uses the ngrams function from NLTK to generate N-grams within the specified range (ngram_range) from the tokenized text in tokenized_corpus. In this case, it will generate bigrams (2-word sequences).

- ngram_freq = Counter(n_grams): This line uses the Counter class to create a dictionary-like object ngram_freq that keeps track of how many times each bigram appears in the corpus.

- top_ngrams = ngram_freq.most_common(top_n): This line uses the most_common method of ngram_freq to get the top_n (e.g., top 5) most frequent bigrams and their counts. It returns a list of tuples, where each tuple contains a bigram and its frequency.

- return top_ngrams: This line returns the list of top N-grams (top_ngrams).

3. Analyzing Fake and Real News Titles:

- fake_news_bigrams = get_ngrams(fake_data['title'], 2, 5): This line calls the get_ngrams function to find the top 5 most frequent bigrams from the titles in the 'title' column of the fake_data DataFrame.

- true_news_bigrams = get_ngrams(real_data['title'], 2, 5): This line does the same for the titles in the 'title' column of the real_data DataFrame, identifying the top 5 most frequent bigrams in real news titles.

4. Creating DataFrame:

- This section creates a DataFrame named bigrams_df to present the results in a tabular format.
  -The DataFrame has two columns:

  - 'Fake News Bi-Grams': This column lists the top 5 bigrams from fake news titles, formatted with each word and its frequency in parentheses (e.g., "breaking news (10)").
    -'True News Bi-Grams': This column lists the top 5 bigrams from real news titles, formatted similarly.
  - The values in each cell are created using list comprehension to iterate through the fake_news_bigrams and true_news_bigrams lists and format the bigrams with their counts.

5. Displaying Results:

- bigrams_df: This line displays the final DataFrame bigrams_df, which allows you to compare the most frequent bigrams used in fake and real news titles.

Overall, this code helps you identify potential patterns in how language is used within fake and real news articles by analyzing the most frequent bigrams (2-word phrases) in their titles.

Output:

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/219da0e9-86c6-4d65-b8bb-2c41213c8b80)

Upon analyzing bi-grams in news titles, distinct patterns emerge between fake and true news datasets.

In fake news titles, the most prevalent bi-gram is 'Donald Trump', appearing 547 times, indicating a notable emphasis on sensationalism or potential political bias. This frequent mention of the former president's name suggests a focus on generating attention or controversy within the fabricated news stories. Following closely behind is the bi-gram 'White House' with a frequency of 268, reinforcing the theme of political intrigue or manipulation within the fabricated narratives.

Conversely, in titles of true news articles, 'White House' emerges as the dominant bi-gram, occurring a staggering 734 times. This prevalence underscores the significance of political coverage and governmental affairs within legitimate news sources. Additionally, 'North Korea' ranks as the second most frequent bi-gram with a frequency of 578, indicating a substantial focus on international relations and geopolitical developments in authentic news reporting.

These findings highlight the divergent thematic focuses between fake and true news titles, with the former often prioritizing sensationalism and political intrigue, while the latter tends to emphasize genuine political events and international affairs. Such insights underscore the importance of discernment and critical analysis when consuming news media, particularly in discerning between fabricated narratives and factual reporting.

```<python>
fake_news_trigrams = get_ngrams(fake_data['title'], 3, 5)
true_news_trigrams = get_ngrams(real_data['title'], 3, 5)

trigrams_df = pd.DataFrame({
    'Fake News Tri-Grams': [f"{trigram[0]} ({trigram[1]})" for trigram in fake_news_trigrams],
    'True News Tri-Grams': [f"{trigram[0]} ({trigram[1]})" for trigram in true_news_trigrams]
})

trigrams_df
```
This code builds upon the previous analysis and focuses on trigrams, which are 3-word phrases, in your fake and real news article titles. Here's a breakdown of what it does:

- Finding Top Trigrams:

  - fake_news_trigrams = get_ngrams(fake_data['title'], 3, 5): This line calls the get_ngrams function you defined earlier. Here, it analyzes the titles in the 'title' column of fake_data, searching for the top 5 most frequent trigrams (3-word sequences).
  -true_news_trigrams = get_ngrams(real_data['title'], 3, 5): This line does the same for the titles in real_data, identifying the top 5 most frequent trigrams used in real news titles.
  
- Creating Trigram DataFrame:

- This section creates a DataFrame named trigrams_df to present the trigram analysis results:
  -Similar to the bigram DataFrame, it has two columns:

  - 'Fake News Tri-Grams': This column lists the top 5 trigrams from fake news titles, formatted with each word and its frequency in parentheses (e.g., "breaking news event (8)").
  - 'True News Tri-Grams': This column lists the top 5 trigrams from real news titles, formatted similarly.
The values are created using list comprehension to iterate through the fake_news_trigrams and true_news_trigrams lists and format the trigrams with their counts.

- Displaying Results:

  - trigrams_df: This line displays the final DataFrame trigrams_df, allowing you to compare the most frequent 3-word phrases used in the titles of fake and real news articles.

By analyzing both bigrams (2-word phrases) and trigrams (3-word phrases), you can gain a deeper understanding of the characteristic language patterns used in different categories of news articles (fake vs. real). This can be helpful in developing algorithms for detecting fake news or understanding the stylistic choices used in these types of content.

Output:

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/6cfeb7c9-7422-48b5-ba87-5b19a68e2cd9)

- The occurrence of phrases like 'Boiler Room EP' and 'Black Lives Matter' within fake news tri-grams suggests a propensity towards sensationalism or subjective interpretation of events. These phrases, often utilized to evoke strong emotions or garner attention, indicate a tendency within fabricated narratives to prioritize dramatic or controversial elements over factual accuracy.

- In contrast, tri-grams found in true news articles containing phrases like 'White House says' and 'Trump says he' reflect a more objective approach to reporting, focusing on statements made by credible sources. By attributing information to authoritative figures or institutions, true news sources aim to provide readers with reliable and verifiable information, thereby upholding journalistic integrity.

- Tri-grams within fake news articles may include phrases that are intentionally misleading or crafted to deceive, as exemplified by the inclusion of 'To Vote For'. Such deceptive practices aim to manipulate readers' perceptions or influence their behavior, highlighting the deceptive nature inherent in fabricated news narratives.

- Conversely, tri-grams present in true news articles often incorporate phrases that offer context or factual information, as illustrated by 'Factbox: Trump on'. By providing readers with additional background or explanatory details, true news sources strive to enhance understanding and clarity surrounding complex issues or events, fostering informed discourse and critical thinking among readers.



