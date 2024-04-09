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

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
```

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





