# Introduction

In today's world, where we use the internet and social media a lot, there's a ton of information being shared all the time. But sometimes, this information isn't accurate, and that can cause problems. Fake news spreads quickly, and it can change how people think and talk about things. To stop this from happening, we can use special computer techniques like Natural Language Processing and Machine Learning. By combining these methods, we can better understand and predict what's true and what's not, helping us tackle this problem more effectively.

ML models have great ability to analyze the data and make the predictions with accurate percentage output. This huge amount of dataflow is controlled by using different enhanced tools and softwares. To find the news is fake and true we need to understand the human language of words where the data should align in this dimensions which will increase the accuracy rate of the NLP algorithms.

The goal of this project is to establish a link between data and human understanding, aiding in the determination of whether data is genuine or false. The dataset employed for training purposes will familiarize the system with commonly used patterns and address missing information. Through an iterative process, the approach is fine-tuned and evaluated on the dataset, resulting in the development of an accurate model that fosters trust by eliminating fake news. The objective is to enhance the accuracy of information in digital media.

# Literature Review
﻿The proliferation of fake information in present day virtual panorama has underscored the important want for powerful detection methods. Various studies have been performed to deal with this pressing trouble, employing various procedures and strategies. This literature assessment offers a top level view of key studies inside the field of faux news detection:

"Fake News Detection on Social Media: A Data Mining Perspective" with the aid of Shu et al. (2017): Shu et al.This study proposes a framework for detecting fake news on social media using data mining techniques. They explore features such as linguistic characteristics, user engagement, and propagation patterns to distinguish between fake and real news.

**Used Logistic Regression and NLP Technique(Term Frequency-Inverse Document Frequency)**

"Fake News Detection: A Deep Learning Approach" by means of Patel et al. (2018): Patel et al. proposed a deep learning-based approach for fake news detection, leveraging techniques such as Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. They demonstrate the effectiveness of their approach on a large dataset of news articles.

**Used Multi Layer Perceptron and NLP Technique(Stance Detection)**

"Detection of Fake News in Social Media Networks" by way of Wang et al. (2018): Wang et al. presented a comprehensive survey of fake news detection techniques in social media networks. They discuss various approaches, including content-based, user-based, and propagation-based methods, highlighting their strengths and limitations.

**Used Classification Algorithms, Python script, and statistical methods.**

"A Survey of Fake News Detection Methods" through Sharma et al. (2019): Sharma et al. provided an extensive survey of fake news detection methods, categorizing them into linguistic, network-based, and hybrid approaches. They discuss the challenges associated with each approach and identify future research directions.

**Survey on all previous approaches (Solutions or Mitigation Techniques)**

"Fake News Detection Using Natural Language Processing" by means of Jin et al. (2020): Jin et al. proposed a fake news detection framework based on Natural Language Processing (NLP) techniques. They extract linguistic features from news articles and use machine learning algorithms to classify them as fake or real.

**Used NLP Techniques and Logistic Regression(Count Vectorizer)**


These studies collectively contribute to advancing the field of fake news detection, offering insights into diverse methodologies and approaches aimed at mitigating the harmful effects of misinformation.

# Dataset Overview
The dataset is a collection of news articles available through the link [Fake News Detection Dataset](https://www.kaggle.com/code/therealsampat/fake-news-detection/input).

  Number of files: 2 
  Total size: 116.37 MB

**Document 1: Fake.csv**

  Size: 62.79MB
  
  Number of rows: 23,481
  
  Number of columns: 4

**Document 2: True.csv**

  Size: 53.58MB
  
  Number of rows: 21,417
  
  Number of columns: 4

# Data Overview

fake_data.head()

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/52a77b10-a4a8-4c7f-9654-2589ce98e188)

fake_data.info()

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/d8374bca-86b1-4370-8011-76fd8286f86b)

real_data.head()

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/e930b673-f3b0-43d1-9cca-51e0ea7f230a)

real_data.info()

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/c8de7c6d-6efa-4e14-8655-7a5fa9637664)

# Data Preprocessing

## Concatenation of Datasets:
The dataset is initially segregated into two CSV files: Fake.csv and Real.csv. The process involves merging them and introducing a target variable: assigning 0 to Fake News and 1 to True/Real News.
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/0d23e9da-fc87-48f3-93a2-e4292a580e55)


## Checking for Null Values:
Upon combining the dataset, it is confirmed that there are no null values present.
![Image for Null Values Check](image_link_here)

## Text Cleaning Process

1. **Noise Removal:**
   Stripped text of formatting, including HTML tags, special characters, and emojis, to focus solely on the textual content.

2. **Lowercasing:**
   Converted all text to lowercase to maintain consistency; for example, "Hello" and "hello" were treated as the same word for analysis.

3. **Punctuation Removal:**
   Discarded punctuation marks from the text, as they typically don't convey significant semantic meaning.

4. **Stopwords Removal:**
   Removed common words like "is", "and", "the", etc., known as stopwords, which often don't contribute much to the text's meaning.

5. **Numbers Removal:**
   Eliminated numerical values or converted them into textual representations, as they might not be relevant for the analysis.

## Preprocessing Steps in NLP

1. **Tokenization:**

***Sentence Tokenization:*** Divided the text into individual sentences.

***Word Tokenization:*** Split the text into individual words or tokens.

2. **Stemming:**
   Reduced words to their base form (e.g., "running" to "run").

3. **Lemmatization:**
    Mapped contextually linked words with similar meanings to a single word (e.g., "better" to "good").

4. **Word2Vec Embeddings:**
    Word2Vec maps words into a dense vector space, capturing semantic meanings based on their context in the corpus.
    Converted sentences to their average word vector representations using a Word2Vec model.


  

# Exploratory Data Analysis
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/69dc4600-b481-449b-bc5e-e1e8dd7a180b)

﻿This bar graph depicts concern frequency, probable in a news article dataset. The x-axis lists topics like "US News" and "Politics", with the y-axis showing their frequency (possibly range of articles). Red and white bars represent two categories (unclear from missing legend), with "US News" and "Politics" being the maximum common subjects universal.

 ![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/4f8d88d2-066f-447f-b065-41eaa2c51fc5)
 
 This pie chart likely depicts the distribution of real and fake data in your dataset. With blue labeled as 52.3% and orange at 47.7%, the data seems fairly balanced between real and fake categories.

 ![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/361e535e-3318-42be-9d84-f4e026651781)
 
 This time series plot visualizes the distribution of fake (light pink) and real (light blue) news articles over time (months). The stacked area chart shows the cumulative number of articles in each category, with color intensity potentially indicating higher volumes of fake news articles compared to real news articles throughout the displayed timeframe.

 ![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/3abb395c-9289-440d-87aa-189f1e4aa320)
 
 Distribution of Article Lengths by Category

 ## Text Processing
 Creating text data for further analysis by by converting it to lowercase, removing punctuation and digits, and potentially feeding it into a word cloud creation process to visualize frequently used words within the fake news content and real news content.

 ![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/7face6f4-b0a2-467d-b4d7-c8481c8efcbe)

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/ccafd248-a823-4ce0-be58-24b4111a80dc)

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/a7967b24-ea1e-499d-ab02-4db9ceb30a6b)

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/d4c04660-f22f-4630-bccd-32636af232ba)

# Model Development
## N GRAM Analysis
N-grams are a fundamental concept in the realm of text analysis, defined as contiguous sequences of n items extracted from a given sample of text or speech. These items can vary depending on the application and context, encompassing letters, words, or even base pairs in the case of genomic analysis. The utility of N-grams extends across various domains, with applications ranging from natural language processing to bioinformatics. In text analysis, N-grams serve as building blocks for tasks such as language modeling, sentiment analysis, and predictive text input. By capturing the sequential relationships between elements within a text, N-grams provide valuable insights into linguistic patterns and structures.

For instance, in the context of language modeling, N-grams are utilized to estimate the probability of encountering a particular sequence of words within a given corpus. This probabilistic approach forms the basis for applications such as text prediction and autocorrection in mobile keyboards and search engines. Moreover, N-grams play a crucial role in the detection of plagiarism and authorship attribution by identifying similarities in textual patterns across documents. By analyzing the frequency and distribution of N-grams, researchers can assess the uniqueness and authenticity of written content, thereby aiding in the preservation of academic integrity and intellectual property rights. In summary, N-grams serve as versatile tools in text analysis, enabling the extraction of meaningful insights from textual data across diverse fields and applications. Through their ability to capture sequential relationships and patterns, N-grams empower researchers and practitioners to unravel the complexities of language and communication in both spoken and written forms.

Overall, identify potential patterns in how language is used within fake and real news articles by analyzing the most frequent bigrams (2-word phrases) in their titles.

### Bi-Gram Analysis

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/9b09a7bf-25be-4e8d-a45d-ca8fc8c0144e)


Upon analyzing bi-grams in news titles, distinct patterns emerge between fake and true news datasets.

In fake news titles, the most prevalent bi-gram is 'Donald Trump', appearing 547 times, indicating a notable emphasis on sensationalism or potential political bias. This frequent mention of the former president's name suggests a focus on generating attention or controversy within the fabricated news stories. Following closely behind is the bi-gram 'White House' with a frequency of 268, reinforcing the theme of political intrigue or manipulation within the fabricated narratives.

Conversely, in titles of true news articles, 'White House' emerges as the dominant bi-gram, occurring a staggering 734 times. This prevalence underscores the significance of political coverage and governmental affairs within legitimate news sources. Additionally, 'North Korea' ranks as the second most frequent bi-gram with a frequency of 578, indicating a substantial focus on international relations and geopolitical developments in authentic news reporting.

These findings highlight the divergent thematic focuses between fake and true news titles, with the former often prioritizing sensationalism and political intrigue, while the latter tends to emphasize genuine political events and international affairs. Such insights underscore the importance of discernment and critical analysis when consuming news media, particularly in discerning between fabricated narratives and factual reporting.

By analyzing both bigrams (2-word phrases) and trigrams (3-word phrases), you can gain a deeper understanding of the characteristic language patterns used in different categories of news articles (fake vs. real). This can be helpful in developing algorithms for detecting fake news or understanding the stylistic choices used in these types of content.

### Tri-Gram Analysis

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/c36c7b23-532f-4074-b2e2-45785e15e2ca)

The occurrence of phrases like 'Boiler Room EP' and 'Black Lives Matter' within fake news tri-grams suggests a propensity towards sensationalism or subjective interpretation of events. These phrases, often utilized to evoke strong emotions or garner attention, indicate a tendency within fabricated narratives to prioritize dramatic or controversial elements over factual accuracy.

In contrast, tri-grams found in true news articles containing phrases like 'White House says' and 'Trump says he' reflect a more objective approach to reporting, focusing on statements made by credible sources. By attributing information to authoritative figures or institutions, true news sources aim to provide readers with reliable and verifiable information, thereby upholding journalistic integrity.

Tri-grams within fake news articles may include phrases that are intentionally misleading or crafted to deceive, as exemplified by the inclusion of 'To Vote For'. Such deceptive practices aim to manipulate readers' perceptions or influence their behavior, highlighting the deceptive nature inherent in fabricated news narratives.

Conversely, tri-grams present in true news articles often incorporate phrases that offer context or factual information, as illustrated by 'Factbox: Trump on'. By providing readers with additional background or explanatory details, true news sources strive to enhance understanding and clarity surrounding complex issues or events, fostering informed discourse and critical thinking among readers.

# Preparing Text for Machine Learning Task
Here's a breakdown of the key functionalities:

### ﻿Text Preprocessing:

It defines a characteristic text_preprocess that converts textual content to lowercase, removes punctuation and non-alphabetic characters, and applies stemming (decreasing phrases to their root form) the use of PorterStemmer. The parallel_preprocessing function leverages more than one CPU cores to correctly technique huge datasets.

### Word2Vec Embeddings: 

It trains a Word2Vec version (get_word2vec_embeddings) to seize semantic relationships between words based totally on their co-occurrence in sentences. The sentence_to_avg_vector characteristic then converts sentences into their common phrase vector illustration using the educated version.

### Data Splitting: 

It splits the records into training, validation, and testing sets (split_data) to teach and evaluate a machine gaining knowledge of version.

### Feature Extraction: 

It gives a feature extract_features that allows selecting between Bag-of-Words (bow) or TF-IDF strategies to convert textual statistics into numerical capabilities suitable for device mastering algorithms.

### Integration of Word2Vec Embeddings: 

It integrates the Word2Vec embeddings as extra features for the version. Each statistics point now has features extracted from both conventional textual content processing methods and the Word2Vec model, probably enriching the model's ability to learn from the facts.


Overall, ﻿Prepares textual content records for machine learing knowledge of through cleaning, transforming it into functions, and incorporating word embeddings to probably enhance the model's performance in classifying fake and real news articles.

# Data Models
We used different classification models to find out the best model and saved the model weights, here is breakdown process:
### ﻿Importing Classifiers and Metrics:

It imports diverse system studying classifiers (Logistic Regression, SVM, Random Forest, and so on.) and evaluation metrics (F1 rating, confusion matrix, ROC curve).

### Classifiers Dictionary:

It creates a dictionary named classifiers that shops different system mastering models with their default parameters.

### Model Evaluation Loop:

It iterates via every classifier within the classifiers dictionary:
Trains the model at the schooling statistics with Word2Vec features (xtrain_w2v, ytrain).
Makes predictions on the checking out information (xtest_w2v).

### Calculates assessment metrics:

F1 rating (harmonic mean of precision and recall)
Confusion matrix (visualizes accurate and incorrect predictions)
ROC curve and Area Under the Curve (AUC) to measure version overall performance.
Stores the results for each version with Word2Vec capabilities in a dictionary named res.

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/a40494ee-119c-44dd-92dc-7c0d3cb5ea9f)

### Results DataFrame:

It converts the res dictionary right into a pandas DataFrame (res_df) for better employer and visualization of the assessment results.

### ROC Curve Visualization:

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/e69fe8e9-83e5-4bf1-aad8-0cd1bfaf6a4f)


It plots ROC curves for each version using their corresponding False Positive Rate (FPR), True Positive Rate (TPR), and AUC (Area Under the Curve) values, taking into consideration a visible assessment of version overall performance.

### Hyperparameter Tuning (Logistic Regression):

It defines hyperparameters to tune for Logistic Regression (log_params).
It uses GridSearchCV to perform hyperparameter tuning, attempting to find the best combination of hyperparameter values that maximize the F1 score on a 5-fold cross-validation technique.
It prints the best hyperparameters located for Logistic Regression.

### Saving the Best Model:

It saves the exceptional appearing Logistic Regression model (after hyperparameter tuning) using joblib for destiny use.

Overall, facilitates perceive the quality machine learning algorithm to know version for classifying fake and real news articles primarily based on Word2Vec embeddings after which refines the chosen version (Logistic Regression) the usage of hyperparameter tuning to potentially improve its overall performance.














