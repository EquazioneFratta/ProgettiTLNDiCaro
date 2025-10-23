import nltk
import pandas as pd
import spacy
from nltk.corpus import wordnet as wn

nlp = spacy.load('it_core_news_md')

# nltk.download('wordnet')
# nltk.download('omw-1.4')

def italian_synonyms(term):
    synsets = wn.synsets(term, lang='ita')
    synonyms = set()
    for syn in synsets:
        for lemma in syn.lemma_names('ita'):
            synonyms.add(lemma)
    return list(synonyms)

def find_synsets_from_definition(definition, target_term=None, score_threshold=0.05):
    doc = nlp(definition.lower())
    keywords = {token.lemma_ for token in doc
                if not token.is_stop and token.is_alpha and len(token.lemma_) > 2}

    search_terms = set(keywords)
    if target_term:
        search_terms.add(target_term)
        search_terms.update(italian_synonyms(target_term))

    candidate_synsets = []
    for term in search_terms:
        candidate_synsets.extend(wn.synsets(term, lang='ita'))

    scored_synsets = []
    for synset in candidate_synsets:
        lemmas_ita = synset.lemma_names('ita')
        wn_doc = nlp(" ".join(lemmas_ita).lower())
        wn_keywords = {token.lemma_ for token in wn_doc if not token.is_stop and token.is_alpha}
        wn_keywords.update(lemmas_ita)

        intersection = len(keywords & wn_keywords)
        union = len(keywords | wn_keywords)
        score = intersection / union if union > 0 else 0

        # Leggero boost se intersezione con iperonimi
        hypernyms = synset.hypernyms()
        hypernyms_lemmas = set()
        for h in hypernyms:
            hypernyms_lemmas.update(h.lemma_names('ita'))

        if len(keywords & hypernyms_lemmas) > 0:
            score += 0.1

        if score >= score_threshold:
            scored_synsets.append((score, synset))

    scored_synsets.sort(key=lambda x: x[0], reverse=True)
    top_synsets = scored_synsets[:3]

    results = []
    for score, synset in top_synsets:
        results.append({
            'synset': synset,
            'synset_name': synset.name(),
            'lemmas_ita': synset.lemma_names('ita'),
            'lemmas_eng': synset.lemma_names('eng'),
            'wn_definition': synset.definition(),
            'score': score,
            'hypernyms': [h.lemma_names('ita') for h in synset.hypernyms()],
            'pos': synset.pos()
        })

    return results

def main():
    df = pd.read_excel('dataset_definizioni_TLN_25.xlsx')
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
            if i > 5:
                break  # Limita a prime 5 definizioni per termine
            print(f"\n Definizione {i}:")
            print(f"   {definition}")

            synset_candidates = find_synsets_from_definition(definition, target_term=term)

            if synset_candidates:
                print(f"\n Synset candidati trovati:")
                for idx, res in enumerate(synset_candidates, 1):
                    print(f"  {idx}. Synset: {res['synset_name']}")
                    print(f"     Definizione WN (inglese): {res['wn_definition']}")
                    print(f"     Lemmi italiani: {', '.join(res['lemmas_ita'])}")
                    # print(f"     Score: {res['score']:.3f}")
                    # if res['hypernyms']:
                    #     print(f"     Iperonimi (lemmi italiani): {', '.join(sum(res['hypernyms'], []))}")
            else:
                print(f"\n Nessun synset trovato")

            term_results.append(synset_candidates)

        results[term] = term_results

if __name__ == "__main__":
    main()
