import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

punctuation_pattern = re.compile(r"[-$<>*?#:';+&/,.\@\(\)!]")


# Lowercase
def lower_case(text):
    return text.lower()


# Remove URLs, emails, numbers, non-ascii characters
def removing_characters(text):
    clean = re.sub(r'http\S+\s', ' ', text)
    clean = re.sub(r'\S+@\S+', ' ', clean)
    clean = re.sub(r'\b\d{10,}\b', ' ', clean)
    clean = re.sub(r'\d+', ' ', clean)
    clean = re.sub(r'[^\x00-\x7f]', ' ', clean)

    return clean


# Remove punctuation
def clean_punctuation(text):
    text = punctuation_pattern.sub(" ", text)
    text = text.replace('－', ' ')

    return text


# Remove extra spaces and new lines
def new_line(text):
    if not isinstance(text, str):
        return text

    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\xa0', ' ')

    return text


# Tokenization
def text_token(text):
    return word_tokenize(text)


# Remove short words and non alphabetic words
def remove_short_words(text):
    return [
        word for word in text
        if word.isalpha() and len(word) > 2
    ]


# Remove stopwords
def remove_stopwords(text):
    return [
        word for word in text
        if word not in stop_words
    ]


# Lemmatization
def lemmatization(text):
    return [
        lemmatizer.lemmatize(word)
        for word in text
    ]


# Join tokens back into sentence
def join_text(text):
    return " ".join(text)



# Complete preprocessing pipeline
def preprocess_resume(text):

    text = lower_case(text)

    text = removing_characters(text)

    text = clean_punctuation(text)

    text = new_line(text)

    text = text_token(text)

    text = remove_short_words(text)

    text = remove_stopwords(text)

    text = lemmatization(text)

    text = join_text(text)

    return text