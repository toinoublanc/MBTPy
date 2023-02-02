import tqdm
import re

def clean_text(text):
    cleaned_text=[]
    for sentence in tqdm(text):
        sentence=sentence.lower()
        sentence=re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+',' ',sentence)
        sentence=re.sub('[^0-9a-z]',' ',sentence)
        cleaned_text.append(sentence)
    return cleaned_text









# import pandas as pd
# import re
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# stop_words = set(stopwords.words("english"))

# def preprocessor(text):
#     text = re.sub(r'http\S+', '', text) # remove URLs
#     text = re.sub(r'@\S+', '', text) # remove username mentions
#     text = re.sub(r'[^\w\s]', '', text) # remove punctuation and special characters
#     text = re.sub(r'\d+', '', text) # remove digits
#     text = text.lower() # convert to lowercase
#     text = " ".join([word for word in word_tokenize(text) if word not in stop_words]) # remove stop words
#     return text

# def preprocessor_pipeline(df):
#     df["posts"] = df["posts"].apply(preprocessor)
#     return df

# if __name__ == "__main__":
#     data = pd.read_csv('../data/raw/mbti_1.csv')
#     preprocessed_data = preprocessor_pipeline(data)
#     print(preprocessed_data.head())








