import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

from tqdm import tqdm  # Import tqdm for progress bar

# Load data
true_data = pd.read_csv('data/True.csv')
fake_data = pd.read_csv('data/Fake.csv')

true_data['label'] = 1
fake_data['label'] = 0

data = pd.concat([true_data, fake_data], ignore_index=True)

# Text cleaning with progress bar
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

cleaned_data = []
for text in tqdm(data['text']):
    cleaned_data.append(clean_text(text))

data['text'] = cleaned_data

# Rest of your code for feature engineering, model training, and evaluation
# ...

# Set the maximum number of words and sequence length
MAX_WORDS = 10000
MAX_SEQ_LENGTH = 300

# Tokenizer setup
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH)


X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, data['label'], test_size=0.2, random_state=42)


model = Sequential()
model.add(Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_SEQ_LENGTH))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

np.save('preprocessed data/X_test.npy', X_test)
np.save('preprocessed data/y_test.npy', y_test)

# Make predictions and save them
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Predict and threshold
np.save('preprocessed data/y_pred.npy', y_pred)

history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate on test data
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

def predict_fake_news(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQ_LENGTH)
    prediction = model.predict(padded_sequence)
    return 'Fake' if prediction < 0.5 else 'Real'

model.save('model/fake_news_detection_model.h5')

# Save history for accuracy and loss
import pickle
with open('model/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
