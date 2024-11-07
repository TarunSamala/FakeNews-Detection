import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model('model/fake_news_detection_model.h5')

MAX_SEQ_LENGTH = 300

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

def predict_fake_news(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQ_LENGTH)
    prediction = model.predict(padded_sequence)
    return 'Fake' if prediction < 0.5 else 'Real'

# Example usage
if __name__ == "__main__":
    text = input("Enter the news to check whether its fake: ")
    result = predict_fake_news(text)
    print(f"Provided news is : {result}")
