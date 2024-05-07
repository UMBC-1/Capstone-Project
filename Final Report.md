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

``` python
fake_data.head()
```

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/52a77b10-a4a8-4c7f-9654-2589ce98e188)

``` python
fake_data.info()
```

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/d8374bca-86b1-4370-8011-76fd8286f86b)

``` python
real_data.head()
```

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/e930b673-f3b0-43d1-9cca-51e0ea7f230a)

``` python
real_data.info()
```

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/c8de7c6d-6efa-4e14-8655-7a5fa9637664)

# Data Preprocessing

## Concatenation of Datasets:
The dataset is initially segregated into two CSV files: **Fake.csv** and **Real.csv**. The process involves merging them and introducing a target variable: assigning 0 to Fake News and 1 to True/Real News.

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

T﻿his bar graph illustrates the distribution of subject frequencies within the dataset, highlighting "US News" and "Politics" as the most prevalent subjects. 

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/4f8d88d2-066f-447f-b065-41eaa2c51fc5)
 
The pie chart illustrates the distribution of real and fake data in the dataset, showcasing a fair balance between the two categories.

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/361e535e-3318-42be-9d84-f4e026651781)
 

The above graph demonstrates the connection between article counts and their release dates. Notably, there is a significant increase in articles during May and June of 2016. Additionally, a spike in articles is observed towards late 2017.

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/3abb395c-9289-440d-87aa-189f1e4aa320)
 
**Distribution of Article Lengths by Category**

## Word Cloud 

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/7face6f4-b0a2-467d-b4d7-c8481c8efcbe)

From the word cloud, we can observe that the most commonly used words in fake news articles seem to be "said", "Donald Trump", "American", "people", "that", "according", "support", "action", "women”,  "Hillary Clinton".

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/ccafd248-a823-4ce0-be58-24b4111a80dc)

From the word cloud, we can observe that the most commonly used words in real  news articles seem to be "said", "Donald Trump", "percent ", "people", "that", "united state ", "support", "action", "wednesday”,  "whitehouse" ,”government”.

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/a7967b24-ea1e-499d-ab02-4db9ceb30a6b)

From the  above visualization , it's evident that articles categorized as true tend to have a greater average word length compared to those categorized as fake. Typically, individuals fabricating information tend to employ numerous articles and prepositions. Conversely, individuals conveying truthful information often exhibit greater articulateness and conciseness in their language use . Therefore, the conclusion drawn from the visualization, indicating that true news articles have a shorter average word length, appears to be valid.

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/d4c04660-f22f-4630-bccd-32636af232ba)

The above graphs reveal that fake texts generally contain more characters and words per article, thereby supporting the hypothesis established by the preceding visualization.

## N GRAM Analysis

**N-grams** are a fundamental concept within the realm of textual content evaluation, described as contiguous sequences of n items extracted from a given pattern of textual content or speech. These objects can range depending at the application and context, encompassing letters, words, or maybe base pairs in the case of genomic analysis. The tool of **N-grams** extends throughout various domain names, with applications ranging from natural language processing to bioinformatics. In textual content analysis, **N-grams** function constructing blocks for tasks such as language modeling, sentiment evaluation, and predictive textual content input. By capturing the sequential relationships between elements within a textual content, N-grams provide precious insights into linguistic patterns and systems.

For example, inside the context of language modeling, **N-grams** are utilized to estimate the chance of encountering a selected sequence of words within a given corpus. This probabilistic approach bureaucracy the premise for programs along with textual content prediction and autocorrection in cellular keyboards and engines like google. Moreover, **N-grams** play a essential function in the detection of plagiarism and authorship attribution by using figuring out similarities in textual styles across files. By analyzing the frequency and distribution of **N-grams**, researchers can check the uniqueness and authenticity of written content material, thereby assisting in the preservation of instructional integrity and highbrow assets rights. In precis, **N-grams** function flexible tools in textual content evaluation, enabling the extraction of significant insights from textual data across various fields and programs. Through their capacity to capture sequential relationships and patterns, **N-grams** empower researchers and practitioners to unravel the complexities of language and communication in both spoken and written forms.

Overall, identify potential patterns in how language is used within fake and real news articles by analyzing the most frequent bigrams (2-word phrases) in their titles.

## Bi-Gram Analysis

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/9b09a7bf-25be-4e8d-a45d-ca8fc8c0144e)


Upon analyzing **bi-gram** in news titles, distinct patterns emerge between fake and true news datasets.

+ In fake news titles, the most prevalent bi-gram is 'Donald Trump', appearing 547 times, indicating a notable emphasis on sensationalism or potential political bias. This frequent mention of the former president's name suggests a focus on generating attention or controversy within the fabricated news stories. Following closely behind is the bi-gram 'White House' with a frequency of 268, reinforcing the theme of political intrigue or manipulation within the fabricated narratives.

* Conversely, in titles of true news articles, 'White House' emerges as the dominant bi-gram, occurring a staggering 734 times. This prevalence underscores the significance of political coverage and governmental affairs within legitimate news sources. Additionally, 'North Korea' ranks as the second most frequent bi-gram with a frequency of 578, indicating a substantial focus on international relations and geopolitical developments in authentic news reporting.

+ ﻿These findings spotlight the divergent thematic focuses among fake and actual news titles, with the former frequently prioritizing sensationalism and political intrigue, while the latter has a tendency to emphasize genuine political activities and worldwide affairs. Such insights underscore the importance of discernment and critical evaluation whilst eating information media, in particular in discerning among fabricated narratives and genuine reporting.

- By studying both bigrams (2-phrase phrases) and trigrams (3-word terms), we will gain a deeper information of the function language styles used in exceptional classes of news articles (faux vs. Actual). This can be useful in developing algorithms for detecting faux news or information the stylistic choices used in those styles of content material.

## Tri-Gram Analysis

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/c36c7b23-532f-4074-b2e2-45785e15e2ca)

Upon analyzing **tri-gram** in news titles, distinct patterns emerge between fake and true news datasets.

- The occurrence of phrases like 'Boiler Room EP' and 'Black Lives Matter' within fake news tri-grams suggests a propensity towards sensationalism or subjective interpretation of events. These phrases, often utilized to evoke strong emotions or garner attention, indicate a tendency within fabricated narratives to prioritize dramatic or controversial elements over factual accuracy.

* In contrast, tri-grams found in true news articles containing phrases like 'White House says' and 'Trump says he' reflect a more objective approach to reporting, focusing on statements made by credible sources. By attributing information to authoritative figures or institutions, true news sources aim to provide readers with reliable and verifiable information, thereby upholding journalistic integrity.

+ Tri-grams within fake news articles may include phrases that are intentionally misleading or crafted to deceive, as exemplified by the inclusion of 'To Vote For'. Such deceptive practices aim to manipulate readers' perceptions or influence their behavior, highlighting the deceptive nature inherent in fabricated news narratives.

* Conversely, tri-grams present in true news articles often incorporate phrases that offer context or factual information, as illustrated by 'Factbox: Trump on'. By providing readers with additional background or explanatory details, true news sources strive to enhance understanding and clarity surrounding complex issues or events, fostering informed discourse and critical thinking among readers.

# Model Development 


## Parallel Processing :

  Parallel processing is used to increase the computational speed of computer systems by performing multiple data-processing operations simultaneously. 
  To expedite text preprocessing, the data is divided into smaller chunks.
  Each chunk undergoes preprocessing tasks independently and simultaneously using multiple CPU cores.
  This parallel processing significantly speeds up the overall preprocessing workflow, especially for large datasets.

## Word2Vec Embeddings: 

Word2Vec is a widely used method in natural language processing (NLP) that allows words to be represented as vectors in a continuous vector space. Word2Vec is an effort to map words to high-dimensional vectors to capture the semantic relationships between words . Words with similar meanings should have similar vector representations, according to the main principle of Word2Vec .

## Average Word Vector Representation:

Average word vector representation is a method used in natural language processing to convert sentences or text sequences into fixed-length numerical vectors. This technique involves first representing each word in the sentence as a word embedding vector, where similar words are closer in vector space. These word embeddings are then averaged element-wise to create a single vector representation for the entire sentence, capturing its semantic meaning based on the meanings of its constituent words. This approach allows for the encoding of variable-length text inputs into a consistent format suitable for machine learning algorithms, facilitating tasks such as text classification, sentiment analysis, and document clustering.

# Model Preparation

Since this task involves classification, the chosen models are classifiers, including **Logistic Regression**, **Linear SVM**, **Random Forest**, **Decision Tree**, and **Gradient Boosting**. Google Colab served as the development environment due to its convenience in importing packages. For testing and validation, 20% and 10% of the dataset were utilized

## Classifier Evaluation:

 Created a function  which evaluates the performance of multiple classifiers using word embeddings (Word2Vec) on a test dataset. It trains each classifier, makes predictions on the test dataset and calculates evaluation metrics and stores them in a dictionary. 

##  Assessment metrics:

1. F1 Score (harmonic mean of precision and recall)
2. Confusion matrix (visualizes accurate and incorrect predictions)
3. ROC curve and Area Under the Curve (AUC) to measure version overall performance.
4. Stores the results for each version with Word2Vec capabilities in a dictionary named **res**.

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/a40494ee-119c-44dd-92dc-7c0d3cb5ea9f)

 From the above output Logistic Regression, Linear SVM, and Random Forest classifiers exhibit strong overall performance, as evidenced by their high F1 scores exceeding 0.96. These classifiers demonstrate effective classification ability, with minimal misclassifications as indicated by their respective confusion matrices. While Decision Tree and Gradient Boosting classifiers show slightly lower F1 scores, around 0.92 and 0.95 respectively, they still demonstrate respectable performance, with slightly higher false positive rates. Overall, the results indicate that Logistic Regression, Linear SVM, and Random Forest classifiers perform exceptionally well in this classification task using Word2Vec embeddings, with Decision Tree and Gradient Boosting classifiers offering competitive performance.

## ROC Curve Visualization:

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/e69fe8e9-83e5-4bf1-aad8-0cd1bfaf6a4f)

The ROC curve is a graphical representation of the True Positive Rate (TPR) against the False Positive Rate (FPR) , we can see the the graph of Logistic regression , Linear svm and Random forest approaching the top-left corner of the plot, suggesting strong discriminative ability between positive and negative instances. 

The AUC represents the area under the ROC curve and summarizes the performance of the classifier across all possible threshold settings.

AUC ranges from 0 to 1, where a higher value indicates better performance. 

The AUC values for Logistic Regression, Linear SVM, and Random Forest are around 0.96, indicating their high True Positive Rate (TPR) and relatively low False Positive Rate (FPR).

# Deploying the Model Using Streamlit

 Streamlit was employed to develop a web application, This includes a text input field and a submit button, enabling users to input text for analysis. Upon submission, the model processes it using pre-trained models and provides real-time predictions on whether the news is fake or real. This interactive functionality not only enhances user accessibility to the model but also offers immediate feedback on the authenticity of news content, demonstrating the practical utility of machine learning models in real-world scenarios.

## Fake News 
add image 
## Real News
add image 

# Limitations

1. Relying solely on Word2Vec embeddings may limit the system's adaptability to evolving forms of fake news that differ significantly from the training data. This could lead to reduced performance in detecting novel deceptive tactics.

2. The computational resources required for training and maintaining sophisticated models, especially at scale for real-world deployment, can be substantial and costly. This includes the need for robust infrastructure and significant financial investments to ensure efficient operation and scalability.

3. Ensuring data quality and addressing biases in training data are essential for model fairness and generalizability. Failure to adequately address these concerns may result in biased predictions and undermine the model's effectiveness in diverse real-world scenarios.

# Future Work

1. Explore transformer-based models such as BERT and GPT to capture contextual information and improve the model's understanding of complex language patterns, potentially enhancing its accuracy in detecting subtle forms of fake news.

2. Experiment with ensemble methods to combine multiple models (e.g., logistic regression, random forest) for enhanced predictive performance and model robustness against various types of deceptive content.

3. Develop mechanisms for real-time monitoring and continuous model updates with new data to adapt to evolving fake news patterns and ensure model relevancy over time. This iterative approach contributes to the ongoing battle against misinformation in the digital landscape.







