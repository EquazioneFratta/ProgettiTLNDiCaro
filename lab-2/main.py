import re
import pandas as pd
import nltk
import itertools
import spacy

from nltk.corpus import stopwords 

nlp = spacy.load("it_core_news_sm")
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     words = text.split()
    
#     stop_words = set(stopwords.words('italian'))
#     words = [w for w in words if w not in stop_words]
        
#     return words
    
def clean_and_lemmatize(text):
    doc = nlp(text.lower())
    return {
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha
    }
    

def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

def main():
    df = pd.read_excel('dataset_definizioni_TLN_25.xlsx')
    # rimuove le ultime due colonne
    df = df.iloc[:, :-2]
    
    term_dict = {
        row[1]: [x for x in row[2:] if pd.notna(x)]  
        for row in df.values
    }   
    
    for termine, definizioni in term_dict.items():
        print(f"{termine}:")
        for i, definizione in enumerate(definizioni, start=1):
            print(f"   {i}. {definizione}")
        print()  

    for term, definitions in term_dict.items():
        print(f"\n🔸 Term: {term}")
        for d1, d2 in itertools.combinations(definitions, 2):
            set1, set2 = clean_and_lemmatize(d1), clean_and_lemmatize(d2)
            overlap = len(set1 & set2) / len(set1 | set2)
            # print(f" - Overlap tra:\n   1) {d1}\n   2) {d2}\n   ➜ {overlap:.2f}")
    
if __name__ == "__main__":
    main()