"""
LAB-1: Integrazione WordNet + Embeddings
Corso: Tecnologie del Linguaggio Naturale 2024/2025
Obiettivo: Valutare la complementarità tra risorse simboliche e distribuzionali
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import wordnet as wn
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ===========================
# PARTE 1: CARICAMENTO RISORSE
# ===========================

class WordNetAnalyzer:
    """Classe per l'analisi di WordNet"""
    
    def __init__(self):
        # Scarica WordNet se necessario
        import nltk
        try:
            wn.synsets('test')
        except:
            nltk.download('wordnet')
            nltk.download('omw-1.4')
    
    def get_synsets(self, word, pos=None):
        """Ottiene tutti i synset per una parola"""
        return wn.synsets(word, pos=pos)
    
    def get_synonyms(self, word, pos=None):
        """Estrae tutti i sinonimi di una parola"""
        synonyms = set()
        for syn in wn.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        synonyms.discard(word)
        return list(synonyms)
    
    def get_hypernyms(self, word, pos=None):
        """Estrae gli iperonimi (categorie superiori)"""
        hypernyms = set()
        for syn in wn.synsets(word, pos=pos):
            for hyper in syn.hypernyms():
                for lemma in hyper.lemmas():
                    hypernyms.add(lemma.name().replace('_', ' '))
        return list(hypernyms)
    
    def get_hyponyms(self, word, pos=None):
        """Estrae gli iponimi (categorie inferiori)"""
        hyponyms = set()
        for syn in wn.synsets(word, pos=pos):
            for hypo in syn.hyponyms():
                for lemma in hypo.lemmas():
                    hyponyms.add(lemma.name().replace('_', ' '))
        return list(hyponyms)
    
    def get_path_similarity(self, word1, word2, pos=None):
        """Calcola similarità basata su percorso in WordNet"""
        synsets1 = wn.synsets(word1, pos=pos)
        synsets2 = wn.synsets(word2, pos=pos)
        
        if not synsets1 or not synsets2:
            return None
        
        max_sim = 0
        for s1 in synsets1:
            for s2 in synsets2:
                sim = s1.path_similarity(s2)
                if sim and sim > max_sim:
                    max_sim = sim
        
        return max_sim if max_sim > 0 else None


class EmbeddingAnalyzer:
    """Classe per l'analisi degli embeddings"""
    
    def __init__(self, embedding_type='glove'):
        """
        Inizializza con embeddings pre-addestrati
        Per semplicità usiamo embeddings generati, ma puoi caricare GloVe/Word2Vec
        """
        self.embedding_type = embedding_type
        self.embeddings = {}
        self._load_embeddings()
    
    def _load_embeddings(self):
        """
        Carica embeddings reali (GloVe/Word2Vec) o simulati
        """
        if self.embedding_type == 'glove':
            try:
                # OPZIONE 1: Scarica automaticamente con gensim
                print("Downloading GloVe embeddings (prima volta ~128MB)...")
                import gensim.downloader as api
                
                model = api.load('glove-wiki-gigaword-100')
                self.embeddings = {word: model[word] for word in model.key_to_index}
                print(f"✓ Caricati {len(self.embeddings)} embeddings GloVe reali")
                
            except Exception as e:
                print(f"⚠ Errore nel caricamento GloVe: {e}")
                print("⚠ Uso embeddings simulati come fallback.")
                self._load_simulated_embeddings()
        else:
            self._load_simulated_embeddings()
    
    def _load_simulated_embeddings(self):
        """Carica embeddings simulati per testing/demo"""
        np.random.seed(42)
        
        # Vocabolario di esempio
        vocab = [
            'bank', 'river', 'shore', 'water', 'finance', 'money', 'loan',
            'dog', 'cat', 'animal', 'pet', 'puppy', 'kitten',
            'car', 'vehicle', 'automobile', 'truck', 'transportation',
            'apple', 'fruit', 'orange', 'banana', 'food',
            'computer', 'technology', 'software', 'hardware', 'laptop'
        ]
        
        # Crea embeddings con struttura semantica
        for i, word in enumerate(vocab):
            self.embeddings[word] = np.random.randn(100) + i * 0.1
        
        # Aggiungi similarità semantiche
        base = np.random.randn(100)
        for w in ['bank', 'river', 'shore', 'water']:
            if w in self.embeddings:
                self.embeddings[w] = base + np.random.randn(100) * 0.3
        
        base = np.random.randn(100)
        for w in ['dog', 'cat', 'animal', 'pet', 'puppy', 'kitten']:
            if w in self.embeddings:
                self.embeddings[w] = base + np.random.randn(100) * 0.3
        
        print(f"✓ Caricati {len(self.embeddings)} embeddings simulati")
    
    def get_embedding(self, word):
        """Ottiene l'embedding di una parola"""
        return self.embeddings.get(word.lower())
    
    def cosine_similarity(self, word1, word2):
        """Calcola similarità coseno tra due parole"""
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        
        if emb1 is None or emb2 is None:
            return None
        
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def most_similar(self, word, top_n=10):
        """Trova le parole più simili"""
        emb = self.get_embedding(word)
        if emb is None:
            return []
        
        similarities = []
        for other_word, other_emb in self.embeddings.items():
            if other_word != word.lower():
                sim = np.dot(emb, other_emb) / (np.linalg.norm(emb) * np.linalg.norm(other_emb))
                similarities.append((other_word, sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]


# ===========================
# PARTE 2: INTEGRAZIONE E ANALISI
# ===========================

class ResourceIntegrator:
    """Integra WordNet ed Embeddings per analisi comparative"""
    
    def __init__(self):
        self.wn_analyzer = WordNetAnalyzer()
        self.emb_analyzer = EmbeddingAnalyzer()
    
    def compare_similarities(self, word_pairs):
        """
        Confronta similarità tra WordNet e Embeddings
        """
        results = []
        
        for w1, w2 in word_pairs:
            wn_sim = self.wn_analyzer.get_path_similarity(w1, w2)
            emb_sim = self.emb_analyzer.cosine_similarity(w1, w2)
            
            results.append({
                'word1': w1,
                'word2': w2,
                'wordnet_sim': wn_sim if wn_sim else 0,
                'embedding_sim': emb_sim if emb_sim else 0
            })
        
        return pd.DataFrame(results)
    
    def analyze_synonym_coverage(self, word):
        """
        Analizza quanto gli embeddings coprono i sinonimi di WordNet
        """
        wn_synonyms = self.wn_analyzer.get_synonyms(word)
        
        coverage = []
        for syn in wn_synonyms:
            emb_sim = self.emb_analyzer.cosine_similarity(word, syn)
            coverage.append({
                'synonym': syn,
                'has_embedding': emb_sim is not None,
                'similarity': emb_sim if emb_sim else 0
            })
        
        return pd.DataFrame(coverage)
    
    def find_semantic_gaps(self, word):
        """
        Identifica gap semantici: parole simili negli embeddings
        ma non in WordNet
        """
        wn_synonyms = set(self.wn_analyzer.get_synonyms(word))
        emb_similar = self.emb_analyzer.most_similar(word, top_n=20)
        
        gaps = []
        for similar_word, sim_score in emb_similar:
            if similar_word not in wn_synonyms and similar_word != word:
                gaps.append({
                    'word': similar_word,
                    'similarity': sim_score,
                    'in_wordnet': similar_word in wn_synonyms
                })
        
        return pd.DataFrame(gaps)


# ===========================
# PARTE 3: VISUALIZZAZIONI
# ===========================

class Visualizer:
    """Classe per le visualizzazioni"""
    
    @staticmethod
    def plot_similarity_comparison(df):
        """Confronto grafico tra similarità WordNet ed Embeddings"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(df['wordnet_sim'], df['embedding_sim'], alpha=0.6)
        axes[0].set_xlabel('WordNet Similarity')
        axes[0].set_ylabel('Embedding Similarity')
        axes[0].set_title('Correlazione tra WordNet ed Embeddings')
        axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.3)
        
        # Calcola correlazione
        valid = df[(df['wordnet_sim'] > 0) & (df['embedding_sim'] != 0)]
        if len(valid) > 1:
            corr, p_value = spearmanr(valid['wordnet_sim'], valid['embedding_sim'])
            axes[0].text(0.05, 0.95, f'Spearman ρ = {corr:.3f}\np-value = {p_value:.3e}',
                        transform=axes[0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Heatmap delle coppie di parole
        pivot = df.pivot_table(values='embedding_sim', 
                               index='word1', 
                               columns='word2', 
                               fill_value=0)
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='coolwarm', 
                   ax=axes[1], cbar_kws={'label': 'Similarity'})
        axes[1].set_title('Heatmap Similarità Embeddings')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_tsne_embeddings(embeddings_dict, words=None):
        """Visualizza embeddings in 2D con t-SNE"""
        if words is None:
            words = list(embeddings_dict.keys())
        
        # Filtra solo parole disponibili
        words = [w for w in words if w in embeddings_dict]
        vectors = np.array([embeddings_dict[w] for w in words])
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1))
        vectors_2d = tsne.fit_transform(vectors)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6)
        
        for i, word in enumerate(words):
            ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]),
                       fontsize=9, alpha=0.8)
        
        ax.set_title('Visualizzazione 2D degli Embeddings (t-SNE)')
        ax.set_xlabel('Dimensione 1')
        ax.set_ylabel('Dimensione 2')
        plt.tight_layout()
        return fig


# ===========================
# PARTE 4: ESECUZIONE ANALISI
# ===========================

def main():
    """Esegue l'analisi completa"""
    
    print("="*60)
    print("LAB-1: INTEGRAZIONE WORDNET + EMBEDDINGS")
    print("="*60)
    
    # Inizializza integratore
    integrator = ResourceIntegrator()
    visualizer = Visualizer()
    
    # 1. ANALISI COPPIE DI PAROLE
    print("\n1. Confronto similarità su coppie di parole...")
    word_pairs = [
        ('dog', 'cat'),
        ('dog', 'puppy'),
        ('bank', 'river'),
        ('bank', 'money'),
        ('car', 'truck'),
        ('apple', 'orange'),
        ('computer', 'laptop')
    ]
    
    sim_df = integrator.compare_similarities(word_pairs)
    print(sim_df.to_string(index=False))
    
    # Visualizza confronto
    fig1 = visualizer.plot_similarity_comparison(sim_df)
    plt.savefig('similarity_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Grafico salvato: similarity_comparison.png")
    
    # 2. ANALISI COPERTURA SINONIMI
    print("\n2. Analisi copertura sinonimi per 'dog'...")
    coverage_df = integrator.analyze_synonym_coverage('dog')
    print(coverage_df.to_string(index=False))
    
    # 3. IDENTIFICAZIONE GAP SEMANTICI
    print("\n3. Gap semantici per 'bank'...")
    gaps_df = integrator.find_semantic_gaps('bank')
    print(gaps_df.head(10).to_string(index=False))
    
    # 4. VISUALIZZAZIONE t-SNE
    print("\n4. Generazione visualizzazione t-SNE...")
    words_to_plot = ['dog', 'cat', 'animal', 'pet', 'bank', 'river', 
                     'money', 'car', 'truck', 'apple', 'fruit']
    fig2 = visualizer.plot_tsne_embeddings(
        integrator.emb_analyzer.embeddings, 
        words_to_plot
    )
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Grafico salvato: tsne_visualization.png")
    
    # 5. STATISTICHE FINALI
    print("\n" + "="*60)
    print("STATISTICHE FINALI")
    print("="*60)
    
    valid_pairs = sim_df[(sim_df['wordnet_sim'] > 0) & (sim_df['embedding_sim'] != 0)]
    if len(valid_pairs) > 1:
        corr, p_val = spearmanr(valid_pairs['wordnet_sim'], 
                                valid_pairs['embedding_sim'])
        print(f"Correlazione di Spearman: {corr:.4f} (p={p_val:.4e})")
    
    coverage_rate = coverage_df['has_embedding'].mean() * 100
    print(f"Copertura embeddings su sinonimi WordNet: {coverage_rate:.1f}%")
    
    print(f"\nParole nel vocabolario embeddings: {len(integrator.emb_analyzer.embeddings)}")
    
    print("\n" + "="*60)
    print("ANALISI COMPLETATA")
    print("="*60)
    
    plt.show()


if __name__ == "__main__":
    main()