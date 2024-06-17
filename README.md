# fake-news-DL
News Classification with Bidirectional RNN
This project showcases a deep learning approach for classifying news articles as either true or false using a Bidirectional RNN model with attention mechanisms. The project encompasses data preprocessing, tokenization, feature extraction using TF-IDF, and the integration of GloVe embeddings for improved model performance.

Project Overview
Objective
To build a robust model that can classify news articles based on their content, identifying them as true or false.

Steps Involved
Data Loading and Preparation

Load news dataset from an Excel file.
Combine text from different columns to form a comprehensive text column.
Text Preprocessing

Clean text data by removing URLs, HTML tags, and non-alphabetic characters.
Apply Named Entity Recognition (NER) and Part-of-Speech (POS) tagging to filter out irrelevant words.
Balancing the Dataset

Balance the dataset by undersampling the majority class to mitigate class imbalance.
Feature Extraction

Convert text data into numerical features using TF-IDF vectorization.
Load pre-trained GloVe embeddings and create an embedding matrix.
Model Building

Define a Bidirectional RNN with attention mechanisms to capture important features from the text.
Use embedding layers initialized with GloVe embeddings to leverage pre-trained word vectors.
Model Training and Evaluation

Train the model using the training data and validate its performance on the test data.
Evaluate the model using accuracy as the primary metric.
Dataset
The dataset comprises news articles with labels indicating whether the news is true (1) or false (0). The data is loaded from an Excel file (data.xlsx).

Preprocessing Techniques
Text Cleaning

Remove URLs, HTML tags, and non-alphabetic characters.
Filter out named entities and stopwords using NER and POS tagging.
TF-IDF Vectorization

Convert the cleaned text into numerical features using TF-IDF vectorization, limiting to the top 2000 features.
GloVe Embeddings

Load pre-trained GloVe embeddings and create an embedding matrix to initialize the embedding layer in the model.
Model Architecture
Embedding Layer: Initialized with GloVe embeddings, non-trainable.
Bidirectional RNN Layer: Using Simple RNN units with regularization and dropout for better generalization.
Attention Layer: To focus on the most relevant parts of the text.
Dense Layers: Fully connected layers with regularization and dropout.
Output Layer: Sigmoid activation for binary classification.
Model Training and Evaluation
Optimizer: Adam optimizer with a learning rate of 0.001.
Loss Function: Binary cross-entropy.
Metrics: Accuracy.
The model is trained for 5 epochs with early stopping to prevent overfitting.

Results
The model achieved an accuracy of approximately test_accuracy on the test dataset, demonstrating its ability to distinguish between true and false news articles effectively.

Usage
To use this model for classifying new articles, preprocess the text data as described, vectorize it using the TF-IDF vectorizer, and pass it through the trained model to get predictions.

Acknowledgements
GloVe: Global Vectors for Word Representation.
TensorFlow and Keras for providing powerful tools to build and train deep learning models.
NLTK for natural language processing tools.
