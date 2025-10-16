import nltk
import pandas as pd
import spacy

from nltk.corpus import wordnet as wn

nlp = spacy.load('it_core_news_md')  

# nltk.download('wordnet')
# nltk.download('omw-1.4')  

def find_synset_from_definition(definition):  # , target_term=None
    # Cerca il synset in WordNet che meglio corrisponde alla definizione.
    
    # Args:
    #     definition: la definizione testuale
    #     target_term: il termine target (opzionale, per verificare correttezza)
    
    # Returns:
    #     dict con synset trovato, punteggio e info
    doc = nlp(definition.lower())
    
    keywords = {token.lemma_ for token in doc 
                if not token.is_stop and token.is_alpha and len(token.lemma_) > 2}
    
    # Se conosci il termine target, cerca tra i suoi synset
    # if target_term:
    #     synsets = wn.synsets(target_term, lang='eng')
    # else:
        # Altrimenti cerca tra tutti i synset delle parole chiave
    synsets = []
    for keyword in keywords:
        synsets.extend(wn.synsets(keyword, lang='ita'))
    
    best_match = None
    best_score = 0
    for synset in synsets:
            # Definizione di WordNet (usa inglese come fallback)
            wn_definition = synset.definition()
            
            # Lemmi italiani del synset
            lemmas_ita = synset.lemma_names('ita')
            
            # Calcola overlap
            wn_doc = nlp(wn_definition.lower())
            wn_keywords = {token.lemma_ for token in wn_doc 
                        if not token.is_stop and token.is_alpha}
            
            # Aggiungi anche i lemmi italiani alle keywords
            wn_keywords.update(lemmas_ita)
            
            intersection = len(keywords & wn_keywords)
            union = len(keywords | wn_keywords)
            score = intersection / union if union > 0 else 0
            
            if score > best_score:
                best_score = score
                best_match = synset
        
    return {
        'synset': best_match,
        'synset_name': best_match.name() if best_match else None,
        'lemmas_ita': best_match.lemma_names('ita') if best_match else [],
        'lemmas_eng': best_match.lemma_names('eng') if best_match else [],
        'wn_definition': best_match.definition() if best_match else None,
        'score': best_score,
        'hypernyms': [h.lemma_names('ita') for h in best_match.hypernyms()] if best_match else [],
        'pos': best_match.pos() if best_match else None
    }
    
def main():
    df = pd.read_excel('dataset_definizioni_TLN_25.xlsx')
    # rimuove le ultime due colonne
    df = df.iloc[:, :-2]
    
    term_dict = {
        row[1]: [x for x in row[2:] if pd.notna(x)]  
        for row in df.values
    }   
    
    results = {}
    
    for term, definitions in term_dict.items():
        print(f"\n{'='*70}")
        print(f"TERMINE: {term.upper()}")
        print(f"{'='*70}")
        
        term_results = []
        
        for i, definition in enumerate(definitions, 1):
            print(f"\n Definizione {i}:")
            print(f"   {definition}")
            
            result = find_synset_from_definition(definition) #  , target_term=term
            
            if result['synset']:
                print(f"\n Synset trovato: {result['synset_name']}")
                print(f"   POS: {result['pos']}")
                print(f"   Definizione WN: {result['wn_definition']}")
                print(f"   Score: {result['score']:.3f}")
                # if result['hypernyms']:
                #     print(f"   Iperonimi: {', '.join(result['hypernyms'][:3])}")
            else:
                print(f"\n Nessun synset trovato")
            
            term_results.append(result)
        
        results[term] = term_results
    
if __name__ == "__main__":
    main()
        

