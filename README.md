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

This combination achieves the highest scores **(0.826 and 0.820)** while balancing feature richness and model updates. However, it comes at the cost of longer training times, which should be considered depending on computational resources.

5.2 LSTM

This preprocessing pipeline is designed to prepare text data for input into an LSTM model. Since LSTMs require numerical input in a consistent format, the pipeline involves tokenization, embedding, and padding. Additionally, it includes training a Word2Vec model on the training data to generate word embeddings.

a) Preprocessing

The preprocessing pipeline is designed to prepare text data for input into an LSTM model. LSTMs require numerical input in a consistent format, so the pipeline involves tokenization, embedding, and padding.

* Tokenization using text_to_word_sequence()
* Embedding : Creates a Word2Vec embedding model from the tokenized training data. Maps each word to a dense vector representation of size 60 (set by vector_size), capturing semantic relationships between words. The we convert each tokenized sentence into a list of vectors using the embed_sentence function.
* Padding: Ensures all sequences are of equal length by padding shorter sequences with zeros (padding='post' adds zeros at the end). Trims longer sequences to a fixed length of 200 tokens (maxlen=200).

b) Training

The model is a sequential binary classification architecture with enhanced complexity for improved performance:

Input Layer: Accepts padded sequences of shape (200, 60) from Word2Vec embeddings.
Masking Layer: Ignores padding values (0) to prevent them from affecting the model.
Bidirectional LSTM: Processes sequential data in both forward and backward directions with 64 units and tanh activation, focusing on long-term dependencies.
Dropout Layer: Prevents overfitting by randomly dropping 30% of the connections.
Dense Hidden Layer: Refines learned features with 32 units and ReLU activation.
Batch Normalization: Stabilizes training and accelerates convergence by normalizing the hidden layer outputs.
Output Layer: Uses a sigmoid function to produce a probability for binary classification.
Compilation: Optimized with adam, using binary_crossentropy for loss and accuracy as a metric.
This architecture integrates bidirectional LSTMs for deeper sequence understanding, along with regularization and normalization to enhance stability and performance.

c) Evaluation

The LSTM model and the word2vec pretrained model were then evaluate on the test dataset achieving **92% accuracy**. 

5.3 Bert

We now use a pre-trained BERT model. It performs tokenization, encapsulation extraction and prepares the data for further processing or training. The resulting encapsulation will be introduced in the previous model. 

a) Preprocessing

We are using AutoTokenizer and TFAutoModel from Hugging Face Transformers library.
We are loading **tokenizer** for the pre-trained bert-tiny model and Bert-tiny as pre-trained BERT model.

1) Tokenization : tokenizer(x) converts each review into token IDs (a list of integers), and ["input_ids"] extracts the token IDs.
2) Embedding: model.predict() runs the tokenized input through the BERT model to obtain embeddings
3) Accessing Hidden State: we are taking the embedding of the first token in the sequence, which is the [CLS] token. After passing the input through the model, the output provides embeddings for each token in the sequence. By selecting [:, 0, :], we are extracting the embedding corresponding to the [CLS] token, which is often used as a summary representation of the entire sequence. This embedding is a fixed-size vector (128 dimensions for BERT Tiny) that represents the entire sentence or sequence.

b) Training

Training is similar to LSTM but we have adapted the structure so that it accepts an input dimension of (n sequences, 128 dimensions)

c) Evaluation

Training shows after 27 epochs an accuary on validation data that does not exceed 80%. 

6️⃣ Model choice

The choice of **LSTM + Word2Vec** is justified for our scenario because it strikes the right balance between speed, simplicity, and performance, especially for tasks with straightforward language patterns or limited data. This combination leverages the strengths of pre-trained Word2Vec embeddings and LSTM's sequential modeling capabilities, making it an efficient and effective choice.

We don't use XGBoost because it doesn't handle sequential or contextual information inherent in text data, as it requires manual feature engineering (e.g., TF-IDF) that often fails to capture semantic and temporal dependencies effectively.

We don't use BERT because, while powerful, it is computationally expensive and slower to train due to its complex transformer-based architecture. For smaller datasets or simpler tasks, it can be overkill, and its performance may not justify the additional training time and resource requirements compared to LSTM + Word2Vec.

7️⃣ API, Deployment & Frontend

- Frontend (Streamlit UI):

The user interacts with a frontend application built using Streamlit, hosted on Streamlit Cloud. It provides an intuitive interface for submitting text reviews and viewing predictions. Please find the repo to the frondend --> https://github.com/speroulakis/Mental_Health_Risks_Front

- Backend (FastAPI on Cloud Run):

The backend is built with FastAPI, exposing RESTful API endpoints.
The application is containerized using Docker and deployed on Google Cloud Run.

- Preprocessing and Machine Learning Model:

Submitted reviews are preprocessed (e.g., tokenized, cleaned) in the backend.
The preprocessed data is passed through our Word2Vec model, which is preloaded in memory.
The model produces embeddings, which are then used by our LSTM model to make predictions.

- Data Flow:

Users submit text reviews via the frontend.
The reviews are sent as HTTP POST requests to the backend.
The backend processes the reviews through the Word2Vec model and the classifier, returning predictions to the frontend in JSON format.
