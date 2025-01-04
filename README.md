Link to the Kaggle Dataset: https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018

Collaborators: @speroulakis @jeangregouse

# PharmaFeel - Sentiment Prediction on drugs review
1️⃣ Overview 

PharmaFeel is a sentiment analysis project designed to process and classify user reviews of medications. By leveraging advanced machine learning techniques, this project aims to identify valuable insights from textual data. The workflow includes the following key steps:

1) Filtering Reviews: Identifying and retaining only the most relevant reviews to ensure data quality.
2) Text Cleaning: Removing noise, standardizing text, and preparing it for analysis.
3) Preprocessing: Tokenizing and vectorizing text data to make it machine-readable.
4) Model Comparison: Evaluating multiple machine learning models to select the best-performing one.
5) Deployment: Deploying the selected model using Docker and FastAPI to create a scalable and efficient backend.
6) Frontend Integration: Developing a user-friendly interface using Streamlit to visualize and interact with predictions.
7) Cloud Hosting: Hosting the application in the cloud for accessibility and reliability.

2️⃣ Data exploration

In the notebook Data_exploration.ipynb, we performed an in-depth analysis to better understand the underlying patterns and structure of the data. This exploration is essential to optimize the data cleaning phase. The primary goal of this step is to identify and remove noise as well as outliers, ensuring the dataset is more reliable and ready for further processing.

First, we observe that most of the review lengths are concentrated between 500 and 1,000 characters. This insight will help us focus on the core reviews and determine the appropriate padding size for deep learning models.
![image](https://github.com/user-attachments/assets/bc6234e3-1564-4757-8dc7-f922d625fc68)

We observe a clear polarity in the rating scores. Most of the reviews are either very positive (8, 9, 10) or very negative (1). Based on this observation, we decided to split the dataset into two categories: low ratings (1, 2, 3) and high ratings (8, 9, 10).
This decision is driven by two main reasons. First, focusing on extremely negative and positive reviews allows the model to better learn and differentiate between very negative and very positive sentiments, avoiding confusion or lack of precision caused by ambiguous intermediate ratings. Second, to ensure the dataset is balanced and the model has enough data to learn effectively, we include sufficient samples in each category by grouping adjacent ratings into broader classes.
![image](https://github.com/user-attachments/assets/2edf242e-ea6a-4936-877c-78cff1d1b1bc)

Some verifications can be done afterwards, such as excluding empty reviews, reviews in other languages, and irrelevant reviews (considering the usefulcount).
Here are the statistical parameters of the usefulcount:

count: 161297.000000

mean: 28.004755

std: 36.403742

min: 0.000000

25%: 6.000000

50%: 16.000000

75%: 36.000000

max: 1291.000000

We observe a high asymmetry in the reviews' usefulcount. 25% of the reviews have a usefulcount between 36 and 1291. These are likely the most popular drugs. Therefore, the usefulcount may be biased by the popularity of the drugs rather than the relevance of the reviews. We will use the usefulcount to filter the data and assume that popular drugs provide relevant reviews. We will exclude unpopular drugs (low usefulcount).

3️⃣ Text cleaning

Text cleaning ensures that the data is in a clean, consistent, and meaningful format that is suitable for the machine learning algorithm, helping the model perform better and learn from the data more effectively. The function preproc helps us to create a new column, 'clean', which standardizes the text data for machine learning. The main steps are:

- Remove Punctuation: It loops through each character in the string st and removes all punctuation marks (using string.punctuation).
- Convert to Lowercase: It then converts all characters in the string to lowercase using casefold().
- Remove Newlines: It replaces newline characters (\n) with spaces.
- Remove Digits: It removes any digits from the string using a list comprehension and isdigit().
- Lemmatization:
It applies lemmatization (reducing words to their root form) to the words in the string:
First, verbs are lemmatized with pos='v'.
Then, nouns are lemmatized using the result of the verb lemmatization (pos='n').
The final lemmatized output is combined into a single string with spaces separating the words.

4️⃣ Dataset balancing

The balance_dataset function aims to balance an imbalanced dataset by equalizing the number of instances for each class in a binary classification problem (e.g., good vs bad sentiment). Some important points to notice:

In machine learning, imbalanced datasets (where one class has significantly more samples than the other) can lead to biased models. A model trained on an imbalanced dataset may:
- Predict the majority class more often, leading to poor performance for the minority class.
- Fail to generalize well to new data, especially if the minority class is underrepresented.

Random sampling is important for several reasons:

- Avoiding Bias: It ensures that the selection of instances for the balanced dataset is not biased by the order or structure of the original data, which could lead to overfitting.
- Improving Generalization: Random sampling helps create a diverse and representative sample of the data, which is critical for generalizing to new, unseen data.
- Ensuring Fairness: By randomly sampling from both classes, we avoid selecting only specific patterns or groups from the data, allowing the model to learn more effectively from all the variations in both classes.

5️⃣ Models

After completing all the preprocessing steps, we proceed with model comparisons. The goal is to evaluate and compare the performance of three different types of models: Machine Learning, LSTM, and Transformers. Each model type requires specific preprocessing tailored to its capabilities and requirements.

For the Machine Learning models, we tested different algorithms, including decision trees, random forests, and gradient boosting models like XGBoost. The reason for choosing machine learning models is that they are often simpler and faster to train compared to more complex models like LSTM and Transformers, making them a good baseline for comparison. Additionally, they tend to perform well on structured data and can be easily fine-tuned.

5.1 XGBoost

XGBoost (Extreme Gradient Boosting) is a highly efficient and scalable implementation of gradient boosting, which is an ensemble machine learning technique.

a) Preprocessing

For XGBoost, TF-IDF is usually the most effective vectorization technique for text classification tasks. It provides a sparse matrix representation of the text, highlighting important terms and reducing the impact of common, less-informative words. This is crucial for models like XGBoost, which perform better with well-defined, informative features.
To streamline the process, TF-IDF will be implemented as part of a pipeline. This pipeline allows for efficient preprocessing and model training, ensuring that the vectorization and classification steps are seamlessly integrated. Moreover, we will optimize its parameters later on through techniques like grid search or random search, ensuring that both the vectorization process and the XGBoost model perform at their best for our dataset.

b) Training

In this step, we focus on training, fine-tuning, and cross-validating our model to identify the optimal parameters for both TF-IDF vectorization and the XGBoost classifier. The process involves using GridSearchCV, a systematic approach to test different combinations of hyperparameters and evaluate their performance.

tfidf__ngram_range:
We test two configurations:
(1,1) to consider only individual words (unigrams).
(1,2) to include both unigrams and pairs of consecutive words (bigrams).

XGB__learning_rate:
We test two learning rates: 0.01 and 0.1.

c) Evaluation

learning_rate=0.1, ngram_range=(1, 2)
Scores: 0.826 (CV 1/2), 0.820 (CV 2/2)
Total Time: ~15-16 minutes per fold.
Performance: Best scores overall, showing that bigrams and a higher learning rate provide the most benefit.

This combination achieves the highest scores (0.826 and 0.820) while balancing feature richness and model updates. However, it comes at the cost of longer training times, which should be considered depending on computational resources.

5.2 LSTM

This preprocessing pipeline is designed to prepare text data for input into an LSTM model. Since LSTMs require numerical input in a consistent format, the pipeline involves tokenization, embedding, and padding. Additionally, it includes training a Word2Vec model on the training data to generate word embeddings.

a) Preprocessing

The preprocessing pipeline is designed to prepare text data for input into an LSTM model. LSTMs require numerical input in a consistent format, so the pipeline involves tokenization, embedding, and padding.

** Tokenization using text_to_word_sequence()
** Embedding : Creates a Word2Vec embedding model from the tokenized training data. Maps each word to a dense vector representation of size 60 (set by vector_size), capturing semantic relationships between words. The we convert each tokenized sentence into a list of vectors using the embed_sentence function.
** Padding: Ensures all sequences are of equal length by padding shorter sequences with zeros (padding='post' adds zeros at the end). Trims longer sequences to a fixed length of 200 tokens (maxlen=200).

b) Training

c) Evaluation

5.3 Bert

a) Preprocessing

b) Training

c) Evaluation

# API
Document main API endpoints here

# Setup instructions
Document here for users who want to setup the package locally

# Usage
Document main functionalities of the package here
