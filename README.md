# 🎱 8 Ball Pool Sentiment Analysis
This project aims to classify the sentiment (negative, neutral, positive) of 8 Ball Pool game reviews on the Play Store. By analyzing the reviews, we can gain insights into the overall user sentiment towards the game and identify areas for improvement. The classification model will help categorize the reviews accurately, enabling developers to make data-driven decisions to enhance the gaming experience.

## 🗂️ Dataset
This project utilizes a dataset of over 90,000 reviews scraped from the Play Store. After performing extensive data cleaning and addressing the issue of unbalanced data, the final dataset consists of a total of 24,000 reviews. This refined dataset will be used for sentiment classification, allowing for accurate analysis of user sentiment towards the 8 Ball Pool game.

## 🛠️ Methodology
The sentiment analysis process involves several key steps, including data preprocessing, labelling, imbalance data handling, feature extraction, and model training. The dataset is preprocessed to remove noise, and tokenize the text. The text data is then transformed into feature vectors using techniques such as TF-IDF and Tokenizer. These features are fed into machine learning and deep learning models such as Logistic Regression, LSTM, and Bi-LSTM for sentiment classification.

## 📈 Results
The sentiment classification models were evaluated using different methodologies and achieved the following accuracy scores:

1. Scheme 1 (LR + 90-10): 89.38%
2. Scheme 2 (LSTM + 80-20): 91.73%
3. Scheme 3 (Bi-LSTM + 70-30): 91.97%

These results demonstrate the effectiveness of the models in accurately classifying the sentiment of 8 Ball Pool game reviews. The highest accuracy was achieved by Scheme 3, which utilized a Bi-LSTM model with a 70-30 train-test split.