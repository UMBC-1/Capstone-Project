# 606 - Capstone Project 

# Project title 

# Enhancing Information Integrity A Machine Learning and NLP Approach to Fake News Detection


## Introduction
  
In today's digital age, the dissemination of information has become increasingly rapid and widespread, thanks to the internet and social media platforms. However, this convenience comes with its own set of challenges, chief among them being the proliferation of fake news. Fake news, often designed to deceive or manipulate readers, poses a significant threat to the integrity of information and the functioning of democratic societies.Addressing the issue of fake news requires a multi-faceted approach, combining technological innovations with human discernment. One promising avenue is the application of machine learning (ML) and natural language processing (NLP) techniques to detect and combat fake news effectively. 

Machine learning algorithms have the ability to analyze vast amounts of data and identify patterns that may indicate the authenticity or credibility of a piece of information. When coupled with NLP, which enables computers to understand and interpret human language, these algorithms become even more powerful tools for discerning the veracity of news articles and other forms of content.

In this project, we aim to leverage the capabilities of ML and NLP to develop a robust fake news detection system. By training our algorithms on a diverse dataset of both real and fake news articles, we can teach them to recognize common characteristics and markers of misinformation. Through iterative refinement and testing, we seek to create a model that can accurately distinguish between reliable and untrustworthy sources. Our ultimate goal is to contribute to the enhancement of information integrity in the digital realm. By providing individuals and organizations with the means to identify and combat fake news, we hope to foster a more informed and resilient society.

Through this project, we not only aim to develop a practical solution to the problem of fake news but also to raise awareness about the importance of critical thinking and media literacy in the digital age. By empowering users to make informed judgments about the information they encounter, we can collectively work towards a more trustworthy and transparent information ecosystem. With this introduction, we lay the groundwork for our exploration of fake news detection through machine learning and NLP. As we delve deeper into the technical aspects of our approach and share our findings and insights, we invite fellow researchers, developers, and enthusiasts to join us in our quest to uphold the integrity of information in the digital era.

## Literature Review

The proliferation of fake news in today's digital landscape has underscored the critical need for effective detection methods. Various studies have been conducted to address this pressing issue, employing diverse approaches and techniques. This literature review provides an overview of key studies in the field of fake news detection:

1. "Fake News Detection on Social Media: A Data Mining Perspective" by Shu et al. (2017):
Shu et al. proposed a data mining framework for detecting fake news on social media platforms. Their approach incorporated features such as linguistic characteristics, user engagement, and propagation patterns to differentiate between fake and genuine news. The study utilized Logistic Regression and Natural Language Processing (NLP) techniques, specifically Term Frequency-Inverse Document Frequency (TF-IDF), to analyze and classify news content.

2. "Fake News Detection: A Deep Learning Approach" by Patel et al. (2018):
Patel et al. introduced a deep learning-based methodology for fake news detection, leveraging Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. Their research showcased the efficacy of deep learning models in discerning fake news from legitimate sources. The study employed Multi-Layer Perceptron and NLP techniques, focusing particularly on stance detection to identify misleading information.

3. "Detection of Fake News in Social Media Networks" by Wang et al. (2018):
Wang et al. conducted a comprehensive survey of fake news detection techniques deployed in social media networks. Their study encompassed content-based, user-based, and propagation-based approaches, analyzing the strengths and limitations of each method. Classification algorithms, Python scripts, and statistical methods were employed to evaluate and validate the effectiveness of the proposed strategies.

4. "A Survey of Fake News Detection Methods" by Sharma et al. (2019):
Sharma et al. provided an extensive survey of fake news detection methodologies, categorizing them into linguistic, network-based, and hybrid approaches. The study elucidated the challenges associated with each method and identified potential avenues for future research and development. The survey encompassed a wide range of solutions and mitigation techniques employed in combating the spread of misinformation.

5. "Fake News Detection Using Natural Language Processing" by Jin et al. (2020):
Jin et al. proposed a fake news detection framework based on Natural Language Processing (NLP) techniques. Their methodology involved extracting linguistic features from news articles and employing machine learning algorithms, particularly Logistic Regression with Count Vectorizer, to classify news content as either fake or real.

These studies collectively contribute to advancing the field of fake news detection, offering insights into diverse methodologies and approaches aimed at mitigating the harmful effects of misinformation.

# Dataset Overview

The dataset comprises a collection of news articles available on Kaggle, accessible via the following link: [Fake News Detection Dataset](https://www.kaggle.com/code/therealsampat/fake-news-detection/input). It consists of two main files:

1. Fake.csv:

- File Size: 62.79MB
* Number of Rows: 23,481
+ Number of Columns: 4

2. True.csv:

- File Size: 53.58MB
* Number of Rows: 21,417
+ Number of Columns: 4

These files contain a structured dataset of news articles, categorized into fake and true news. The data provides valuable resources for researchers and practitioners interested in studying and developing techniques for fake news detection.

# Workflow of the Project
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/a9d22fd4-9977-40c7-a853-02475ee9a13e)

1. Data Acquisition and Understanding

Dataset Selection: Identify the data source relevant to your project's goals. This data can come from various sources like internal databases, web scraping, APIs, or public datasets.
Data Understanding: Explore and analyze the initial data to grasp its characteristics. This includes data types, presence of missing values, outliers, and potential inconsistencies.

2. Data Cleaning and Preprocessing

Missing Value Handling: Address missing data points using techniques like mean/median imputation, deletion, or carrying forward/backward values based on context.
Outlier Treatment: Identify and handle outliers that might skew your analysis. This may involve capping outliers to a specific range, winsorizing, or removal depending on the situation.
Data Cleaning: Address inconsistencies like typos, formatting errors, or special characters to ensure data quality.
Normalization/Scaling: Normalize or scale features (columns) in your data to a common range to prevent specific features from dominating the model training process.
Dimensionality Reduction (Optional): If you have a high number of features, consider dimensionality reduction techniques like Principal Component Analysis (PCA) to reduce the feature space while retaining maximum information.

3. Feature Engineering and NLP Tasks

Feature Engineering: Create new features from existing ones that might be more informative for your models. This can involve feature creation, interaction terms, or binning categorical features.
Natural Language Processing (NLP Tasks): If your data is text-based, perform NLP tasks like tokenization (splitting text into words), stemming (reducing words to their root form), or lemmatization (converting words to their dictionary form) to prepare the text data for modeling.

4. Feature Extraction

Select a subset of relevant features from your data that will be used by the machine learning models for training. Feature selection techniques like correlation analysis, chi-square tests, or feature importance scores can help identify the most informative features.

5. Model Training and Selection

Data Splitting: Divide your cleaned and preprocessed data into two sets: a training set (used to train the models) and a testing set (used to evaluate the models' performance on unseen data).
Algorithm Selection: Choose one or more machine learning algorithms suitable for your problem type. The flowchart provides examples like Logistic Regression (classification problems), Support Vector Machine (classification or regression), Decision Tree (classification or regression), and Random Forest (classification or regression).
Model Training: Train each chosen model on the training set. The model learns patterns and relationships within the data to make predictions.

6. Model Evaluation

Evaluation Metrics: Evaluate the performance of the trained models on the testing set using metrics like accuracy, precision, recall, F1 score, or other relevant metrics depending on your problem (classification vs. regression). This helps you understand how well the model generalizes to unseen data.
Model Selection: Based on the evaluation metrics, choose the model with the best performance for your specific needs.

7. Model Deployment

Web Application Development: Develop a web application to deploy your chosen model. This allows users to interact with the model and obtain predictions based on new input data.

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

-df['date'] = pd.to_datetime(df['date'],errors='coerce'): This line converts the 'date' column in your DataFrame df into a proper datetime format using the pandas.to_datetime function. The errors='coerce' argument ensures that any non-convertible dates are set to NaT (Not a Time) instead of raising errors.

2. Creating the Timeline Data:

-timeline_data = df.groupby([df['date'].dt.to_period('M'), 'category']).size().unstack().fillna(0): This part creates a new DataFrame named timeline_data that summarizes the data by month and category:
  -df.groupby([df['date'].dt.to_period('M'), 'category']): This groups the data in df by two factors: the month ('M') extracted from the datetime column         
  (df['date'].dt.to_period('M')) and the category ('category').
  -size(): This calculates the count of entries within each group, essentially giving you the number of real/fake news articles for each month.
  -unstack(): This transforms the grouped data into a DataFrame with months as rows and categories ('category') as columns. The resulting DataFrame shows the         monthly counts of real and fake news articles.
  -fillna(0): This fills any missing values (months with no articles) in the DataFrame with 0, ensuring a complete timeline for visualization.

Importance of Timeline Conversion:

Converting the date to a timeline format is crucial for creating the desired visualization:

-Time-based Analysis: By converting the date to a monthly period, you can analyze and visualize how the number of real and fake news articles changes over time. Looking at raw dates wouldn't provide a clear picture of these trends.

*Stacked Area Chart: The code uses timeline_data.plot(kind='area',stacked=True) to create a stacked area chart. This chart effectively portrays the volume of real and fake news articles (represented by areas) throughout the months. Without the timeline conversion, such an informative visualization wouldn't be possible.

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
-fake = " ".join(article for article in fake_data["text"]): This line merges all the text content from the "text" column in your "fake_data" DataFrame into a single string named "fake". This combines the text from all fake news articles.

2. Preprocessing Function:
-def preprocess_text(text):: This defines a function named preprocess_text that takes a text string as input.
Text Cleaning Steps:
-text = text.lower(): This line converts all characters in the text to lowercase for consistency.
text = re.sub(r'[^\w\s]', '', text): This uses regular expressions (re.sub) to remove all characters except alphanumeric characters (\w) and whitespace (\s). This eliminates punctuation marks from the text.
-text = re.sub(r'\d+', '', text): Another regular expression removes digits (\d+) from the text, focusing on the words themselves.

3. Cleaning Applied:
-fake_cleaned = preprocess_text(fake): This line applies the preprocess_text function to the combined text string "fake", cleaning it by removing punctuation and digits.

5. Output:
-fake_cleaned[:300]: This line displays the first 300 characters of the cleaned text "fake_cleaned", allowing you to see a sample of the processed text.

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


