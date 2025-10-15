import re
import pandas as pd
import nltk
import itertools
import spacy
import random

from nltk.corpus import stopwords 

nlp = spacy.load('it_core_news_md')  

def preprocess_lexical_overlap(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('italian'))
    return {w for w in words if w not in stop_words}
    
# Lemmatizzazione: Riduce le parole alla loro forma base 
# (es. "mangiato", "mangiando", "mangia" → "mangiare"), 
# catturando meglio la similarità semantica
# non la userei per calcolare la sovrapposizione lessicale
def preprocess_semantic_overlap(text):
    doc = nlp(text.lower())
    # Lemmatizza e mantieni solo parole significative
    return {
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha
    }

def main():
    df = pd.read_excel('dataset_definizioni_TLN_25.xlsx')
    # rimuove le ultime due colonne
    df = df.iloc[:, :-2]
    
    term_dict = {
        row[1]: [x for x in row[2:] if pd.notna(x)]  
        for row in df.values
    }   
    
    # for termine, definizioni in term_dict.items():
    #     print(f"{termine}:")
    #     for i, definizione in enumerate(definizioni, start=1):
    #         print(f"   {i}. {definizione}")
    #     print()  

    # Fase 3 - sovrapposizione lessicale
    print("Sovrapposizione lessicale:")
    for term, definitions in term_dict.items():
        print(f"Termine: {term}")
    
        d1, d2 = random.sample(definitions, 2)
        set1 = preprocess_lexical_overlap(d1)
        set2 = preprocess_lexical_overlap(d2)
        overlap = len(set1 & set2) / len(set1 | set2) 
        print(f" - Overlap tra:\n   1) {d1}\n   2) {d2}\n  = {overlap:.2f} \n")
    
    # Fase 3 - sovrapposizione semantica
    print("Sovrapposizione semantica:")
    for term, definitions in term_dict.items():
        print(f"Termine: {term}")
        
        # da capire quale dei 2 usare
        # Metodo 1: Sovrapposizione sui lemmi (baseline semantico)
        d1, d2 = random.sample(definitions, 2)
        set1 = preprocess_semantic_overlap(d1)
        set2 = preprocess_semantic_overlap(d2)
        overlap = len(set1 & set2) / len(set1 | set2)
                
        # Metodo 2: Similarità vettoriale con spaCy (più robusto)
        doc1 = nlp(d1)
        doc2 = nlp(d2)
        vector_similarity = doc1.similarity(doc2)
                
        print(f" - Overlap tra:\n   1) {d1}\n   2) {d2}")
        print(f"   ➜ Lemma overlap: {overlap:.2f}")
        print(f"   ➜ Vector similarity: {vector_similarity:.2f} \n")
    
    
    # Fase 4 - similiarità lessicale (simlex)
    # "dopo aver applicato tecniche di stopword removal, LEMMATIZZAZIONE, e norma-lizzazione."
    # uso quindi preprocess_semantic_overlap
    print("Similarità lessicale (SimLex):")
    for term, definitions in term_dict.items():
        
        simlex_scores = []
    
        for d1, d2 in itertools.combinations(definitions, 2):
            set1 = preprocess_semantic_overlap(d1)
            set2 = preprocess_semantic_overlap(d2)
            
            common_terms = len(set1 & set2)
            total_terms = len(set1 | set2)
            simlex = common_terms / total_terms 
            
            simlex_scores.append(simlex)
        
        avg_simlex = sum(simlex_scores) / len(simlex_scores)
        # simlex_by_term[term] = avg_simlex
        print(f"{term} = {avg_simlex:.3f}")

    # Fase 4 - Similarità semantica (SimSem)
    print("\nSimilarità semantica (SimSem):")

    simsem_by_term = {}
    for term, definitions in term_dict.items():   
        simsem_scores = []
        
        for d1, d2 in itertools.combinations(definitions, 2):
            doc1 = nlp(d1)
            doc2 = nlp(d2)   
            simsem = doc1.similarity(doc2)
            simsem_scores.append(simsem)
        
        avg_simsem = sum(simsem_scores) / len(simsem_scores)
        simsem_by_term[term] = avg_simsem
        print(f"{term} = {avg_simsem:.3f}")
    
if __name__ == "__main__":
    main()