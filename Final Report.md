# Introduction

The modern technology era of internet and social media platfomrs the dataflow is increased and growing day-by-day. Even though there are lot of changes and challenges have to be faced because of the incorrect facts and figures. The fake news will spread rapidly and attract people mindset which will further change the readers thoughts and future discussions. To eradicate this, using NLP techniques and ML techniques wil be a great solution and innovate approach to solve this issue. To face this we require the combinational approach which helps to get highest accuracy with accurate predictional output.

ML models have great ability to analyze the data and make the predictions with accurate percentage output. This huge amount of dataflow is controlled by using different enhanced tools and softwares. To find the news is fake and true we need to understand the human language of words where the data should align in this dimensions which will increase the accuracy rate of the NLP algorithms.

The aim of this project is to establish a connection between the data and human which helps to understand human configurations to test the data is fake ot true. The dataset used to train the data is going to teach the commonly used characters and the informations whcih is missed. The ieratetive approach is refined and tested on the data which able to create a new model and gives accuarte results that creates new trustworthy source of information by eradicating the fake news. The goal is to increase the accurate information in this digital media. To combat the fake news we need to use this innovative approachable technologies more and handle in efficient manner

﻿Through this undertaking, we no longer handiest aim to broaden a practical option to the hassle of fake information however additionally to elevate recognition about the importance of essential questioning and media literacy within the digital age. By empowering customers to make informed judgments approximately the records they come across, we will together work in the direction of a more truthful and transparent information surroundings. With this creation, we lay the basis for our exploration of faux news detection via machine studying and NLP. As we delve deeper into the technical elements of our method and percentage our findings and insights, we invite fellow researchers, developers, and enthusiasts to join us in our quest to uphold the integrity of facts in the virtual generation

# Literature Review
﻿The proliferation of fake information in present day virtual panorama has underscored the important want for powerful detection methods. Various studies have been performed to deal with this pressing trouble, employing various procedures and strategies. This literature assessment offers a top level view of key studies inside the field of faux news detection:

"Fake News Detection on Social Media: A Data Mining Perspective" with the aid of Shu et al. (2017): Shu et al. Proposed a statistics mining framework for detecting faux information on social media platforms. Their approach incorporated features along with linguistic traits, person engagement, and propagation styles to differentiate between faux and true news. The look at applied Logistic Regression and Natural Language Processing (NLP) strategies, specially Term Frequency-Inverse Document Frequency (TF-IDF), to research and classify information content.

﻿"Fake News Detection: A Deep Learning Approach" by means of Patel et al. (2018): Patel et al. Added a deep mastering-primarily based technique for faux information detection, leveraging Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. Their research showcased the efficacy of deep mastering fashions in discerning faux information from legitimate resources. The study employed Multi-Layer Perceptron and NLP strategies, focusing specifically on stance detection to discover misleading statistics.

﻿"Detection of Fake News in Social Media Networks" by way of Wang et al. (2018): Wang et al. Conducted a complete survey of fake information detection techniques deployed in social media networks. Their take a look at encompassed content-primarily based, person-based, and propagation-primarily based processes, analyzing the strengths and barriers of each method. Classification algorithms, Python scripts, and statistical methods were employed to assess and validate the effectiveness of the proposed techniques.

﻿"A Survey of Fake News Detection Methods" through Sharma et al. (2019): Sharma et al. Supplied an in depth survey of faux information detection methodologies, categorizing them into linguistic, community-based, and hybrid methods. The study elucidated the demanding situations related to each technique and identified potential avenues for future studies and improvement. The survey encompassed a extensive variety of solutions and mitigation strategies hired in combating the unfold of incorrect information.

﻿"Fake News Detection Using Natural Language Processing" by means of Jin et al. (2020): Jin et al. Proposed a faux information detection framework primarily based on Natural Language Processing (NLP) techniques. Their methodology concerned extracting linguistic features from news articles and employing device studying algorithms, particularly Logistic Regression with Count Vectorizer, to classify news content material as both fake or real.

These studies collectively contribute to advancing the field of fake news detection, offering insights into diverse methodologies and approaches aimed at mitigating the harmful effects of misinformation.

# Dataset Overview
﻿The dataset contains a set of news articles available on Kaggle, on hand via the subsequent link: Fake News Detection Dataset. It consists of predominant documents:

### Fake.csv:

- File Size: 62.79MB
- Number of Rows: 23,481
- Number of Columns: 4

### True.csv:

- File Size: 53.58MB
- Number of Rows: 21,417
- Number of Columns: 4

These documents contain a based dataset of information articles, labeled into fake and proper news. The information offers precious resources for researchers and practitioners interested in analyzing and growing techniques for fake information detection.

Workflow of the Project

# Workflow of the Project
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/4101cf1e-f123-4885-b275-b6d29b432b42)


# New Developments
﻿In the realm of text analysis, the mixing of natural language processing (NLP) strategies during pre-processing is important for extracting significant insights from textual information. Techniques which includes tokenization, stop phrase removal, and lemmatization or stemming are normally hired to put together the textual content for analysis. However, to delve deeper into the semantic relationships among phrases, leveraging superior strategies like Word2Vec embedding proves valuable.

﻿Word2Vec allows the transformation of phrases into dense vectors, taking pictures semantic similarities and nuances in which means. By representing phrases in a non-stop vector space, Word2Vec enables obligations together with word similarity and analogy, improving the depth of evaluation. Moreover, to deal with the computational challenges posed by massive datasets, parallel processing techniques can be hired. By dispensing the workload across a couple of processors or cores, parallel processing notably improves efficiency and reduces processing time.

In addition to leveraging sophisticated techniques, supplying a user-pleasant interface is critical for facilitating get admission to to the analysis tools. Streamlit, a powerful Python library, offers a convenient answer for developing interactive internet programs with minimum effort. By integrating Streamlit into the evaluation pipeline, users can effects have interaction with the records, visualize results, and explore insights in real-time.

In precis, the synergy among NLP strategies, Word2Vec embedding, parallel processing, and Streamlit interface not only complements the efficiency of textual content analysis however additionally empowers users to uncover deeper insights from textual records readily.

# Data Preprocessing
﻿In the process of preparing our facts for evaluation, we began through merging two distinct datasets: one comprising real data and the opposite containing fake facts. By consolidating those datasets, we aimed to create a complete corpus that encompasses a diverse range of textual content.

Subsequently, we brought a goal variable to every dataset to facilitate type tasks. Specifically, we assigned a fee of 1 to times originating from the real records, signifying their authenticity, even as times sourced from the faux statistics had been labeled with a fee of zero.

To make certain the integrity of our dataset and hold information fine, we conducted thorough records cleansing strategies. This blanketed figuring out and addressing any null or missing values present inside the blended dataset. By meticulously doing away with or imputing missing values, we fortified the robustness of our dataset, thereby improving the reliability and accuracy of next analyses.

# Understanding About Dataset
fake_data.head()

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/52a77b10-a4a8-4c7f-9654-2589ce98e188)

fake_data.info()

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/d8374bca-86b1-4370-8011-76fd8286f86b)

real_data.head()

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/e930b673-f3b0-43d1-9cca-51e0ea7f230a)

real_data.info()

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/c8de7c6d-6efa-4e14-8655-7a5fa9637664)

### Combined Data
![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/0d23e9da-fc87-48f3-93a2-e4292a580e55)

### Null Values

![image](https://github.com/UMBC-1/Capstone-Project/assets/57500152/7758d3e0-303c-4c64-94d4-274b5d167926)

This concludes there are no null values.

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


Overall, ﻿Prepares textual content records for machine learing knowledge of through cleansing, transforming it into functions, and incorporating word embeddings to probably enhance the model's performance in classifying fake and real news articles.

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














