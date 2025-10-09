import re
import pandas as pd
import nltk

from nltk.corpus import stopwords 

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    
    stop_words = set(stopwords.words('italian'))
    words = [w for w in words if w not in stop_words]
        
    return words

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
    
    
if __name__ == "__main__":
    main()