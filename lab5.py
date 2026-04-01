import pandas as pd
from tqdm import tqdm
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


def process_text(text):
    text = re.sub(r"[^\w\s]", "", text) #амена подобных символов на пустую строку
    text = re.sub(r"\s+", " ", text)#повторяющие пробелы на одинарные
    text = re.sub(r"\d", '', text)#даляем цифры
    text = word_tokenize(text)#создает массив
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
    text = [t for t in text if t not in stopwords_set]
    lematizer = WordNetLemmatizer()
    text = [lematizer.lemmatize(t) for t in text]
    text = [t for t in text if t not in stopwords_set]
    return ' '.join(text)
    # print(text)

data = pd.read_csv('reviews.csv')
tqdm.pandas()
data['label'] = data['sentiment'].progress_apply(lambda label: 1 if label == 'positive' else 0)

data['processed'] = data['review'].progress_apply(process_text)
print(data)

data[['processed', 'label']].to_csv('reviews_processed.csv', index=False, header=True)
