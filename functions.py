from tqdm import tqdm
import re


def clean_text(text):
    cleaned_text=[]
    for sentence in tqdm(text):
        sentence=sentence.lower()
        sentence=re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+',' ',sentence)
        sentence=re.sub('[^0-9a-z]',' ',sentence)
        cleaned_text.append(sentence)
    return cleaned_text


if __name__ == '__main__':
    clean_text()


