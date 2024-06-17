import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Layer, Input, dot, Activation, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
import gensim.downloader as api
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function for text preprocessing with NER and POS tagging
def preprocess_text_with_ner_pos(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Perform Part-of-Speech (POS) tagging
    pos_tags = pos_tag(tokens)
    # Perform Named Entity Recognition (NER)
    ner_tags = ne_chunk(pos_tags)
    # Filter out entities and stopwords
    custom_stopwords = set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
                            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
                            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
                            'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o',
                            're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
                            'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
                            'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
                            'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
    filtered_words = [word for (word, tag) in pos_tags if tag != 'NNP' and word.lower() not in custom_stopwords]
    text = " ".join(filtered_words)
    return text.strip()

# Load the data and concatenate text columns
df = pd.read_excel('data.xlsx')
df = df[(df['Eng_Trans_Statement'] != '') & (df['Eng_Trans_News_Body'] != '')]
df['text'] = df['Eng_Trans_Statement'] + ' ' + df['Eng_Trans_News_Body']
df['text'].fillna('', inplace=True)
df_true = df[df['Label'] == 1]
df_false = df[df['Label'] == 0]
sample_size = min(len(df_true), len(df_false))  # Limit the sample size to the smaller class
df_false_sampled = df_false.sample(n=sample_size, replace=True, random_state=42)  # Sample with replacement
df_balanced = pd.concat([df_true, df_false_sampled])
df_balanced = shuffle(df_balanced, random_state=42)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)  # Removing URLs
    text = BeautifulSoup(text, 'lxml').get_text()  # Removing HTML tags
    text = re.sub("\S*\d\S*", "", text).strip()  # Removing words with digits
    text = re.sub('[^A-Za-z]+', ' ', text)  # Removing non-alphabetic characters
    return text.strip()

# Apply preprocessing with NER and POS tagging
df_balanced['processed_text'] = df_balanced['text'].apply(preprocess_text)
df_balanced['processed_text'] = df_balanced['processed_text'].apply(preprocess_text_with_ner_pos)

# Tokenization and TF-IDF vectorization
max_words = 2000
tfidf = TfidfVectorizer(max_features=max_words)
X = tfidf.fit_transform(df_balanced['processed_text'])
y = df_balanced['Label']

# Convert sparse matrix to NumPy array
X = X.toarray()

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load GloVe embeddings
glove_vectors = api.load("glove-wiki-gigaword-100")

# Create embedding matrix
embedding_matrix = np.zeros((max_words, 100))
for word, i in tfidf.vocabulary_.items():
    if i >= max_words:
        continue
    if word in glove_vectors:
        embedding_matrix[i] = glove_vectors[word]
# Import Bidirectional from keras.layers
from tensorflow.keras.layers import Bidirectional, SimpleRNN

# Define input layer
input_layer = Input(shape=(X_train.shape[1],))

# Embedding layer
embedding_dim = 100
embedding_layer = Embedding(input_dim=max_words, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False)(input_layer)

# Bidirectional Simple RNN layer
bidirectional_rnn_layer = Bidirectional(SimpleRNN(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), dropout=0.5))(embedding_layer)

# Attention layer
attention = dot([bidirectional_rnn_layer, bidirectional_rnn_layer], axes=[2, 2])
attention = Activation('softmax')(attention)
attention = dot([attention, bidirectional_rnn_layer], axes=[2, 1])
attention_output = concatenate([bidirectional_rnn_layer, attention], axis=-1)
attention_output = Dense(100, activation='tanh')(attention_output)

# Dense layers
dense_layer = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(attention_output)
dropout_layer = Dropout(0.5)(dense_layer)
output_layer = Dense(1, activation='sigmoid')(dropout_layer)

# Create the model
bidirectional_rnn_model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
bidirectional_rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
print(bidirectional_rnn_model.summary())

# Train the model
history = bidirectional_rnn_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), callbacks=[EarlyStopping(patience=3)])

# Evaluate the model
loss, accuracy = bidirectional_rnn_model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)