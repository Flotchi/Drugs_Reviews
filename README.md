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



# API
Document main API endpoints here

# Setup instructions
Document here for users who want to setup the package locally

# Usage
Document main functionalities of the package here
