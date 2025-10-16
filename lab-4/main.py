import pandas as pd

# # 2. EMBEDDING
# - TF-IDF (baseline)
# - Word2Vec / Doc2Vec
# - Sentence-BERT (più moderno)
# - OpenAI embeddings (se hai accesso)

# # 3. RIDUZIONE DIMENSIONALE
# - PCA
# - t-SNE
# - UMAP (consigliato, molto efficace)

# # 4. CLUSTERING
# - K-Means
# - DBSCAN
# - HDBSCAN (ottimo per topic modeling)
# - Gaussian Mixture Models

# # 5. TOPIC MODELING
# - LDA (Latent Dirichlet Allocation)
# - NMF (Non-negative Matrix Factorization)
# - BERTopic (stato dell'arte)

import re
import numpy as np
from collections import Counter

# Librerie NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def preprocess_text(text):
    
    stop_words = set(stopwords.words("italian"))       
    lemmatizer = WordNetLemmatizer()
    # stemmer = PorterStemmer()
    
    text = text.lower()
    
    # URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # HTML
    text = re.sub(r'<.*?>', '', text)
    # Email
    text = re.sub(r'\S+@\S+', '',  text)
    # Mantiene solo lettere, numeri e spazi
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    tokens = word_tokenize(text)
            
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization o Stemming, credo non si possano usare entrambi insieme
    tokens =[lemmatizer.lemmatize(word) for word in tokens]
    # tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)
    
# def get_vocabulary_stats(self, texts):
#     """
#     Calcola statistiche sul vocabolario
    
#     Args:
#         texts: lista di testi preprocessati
    
#     Returns:
#         Dizionario con statistiche
#     """
#     all_words = []
#     doc_lengths = []
    
#     for text in texts:
#         if text:
#             words = text.split()
#             all_words.extend(words)
#             doc_lengths.append(len(words))
    
#     word_freq = Counter(all_words)
    
#     return {
#         'total_documents': len(texts),
#         'non_empty_documents': len([t for t in texts if t]),
#         'total_words': len(all_words),
#         'unique_words': len(word_freq),
#         'avg_doc_length': np.mean(doc_lengths) if doc_lengths else 0,
#         'median_doc_length': np.median(doc_lengths) if doc_lengths else 0,
#         'most_common_words': word_freq.most_common(20)
#     }


def main():
    
    df = pd.read_csv('ML-Arxiv-Papers.csv')
    print(f"Colonne disponibili: {list(df.columns)}")
    
    preprocessed_abstracts = []
    for abstract in df.iloc[:, 3]:
        processed = preprocess_text(abstract)
        preprocessed_abstracts.append(processed)

    # 1. Embedding model (puoi cambiare modello e parametri)
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # 2. Creazione modello BERTopic
    topic_model = BERTopic(embedding_model=embedding_model, min_topic_size=30)

    # 3. Fits topics su array di testo preprocessato
    topics, probs = topic_model.fit_transform(preprocessed_abstracts)

    # 4. Informazioni sui topic trovati
    print(topic_model.get_topic_info())
    print("\ntopics :",topics)
    
if __name__ == "__main__":
    main()
