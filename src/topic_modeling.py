"""
Core topic modeling using LDA (Latent Dirichlet Allocation).
"""

from gensim.models import LdaModel
from gensim.corpora import Dictionary
from . import utils

class TopicModeler:
    def __init__(self, num_topics=40, random_state=42):
        self.num_topics = num_topics
        self.random_state = random_state
        self.dictionary = None
        self.model = None
        
    def prepare_corpus(self, docs, additional_stopwords=None):
        """Prepare text corpus for topic modeling."""
        # Preprocess documents
        processed_docs = [utils.preprocess_text(doc) for doc in docs]
        processed_docs = utils.remove_custom_stopwords(processed_docs, additional_stopwords)
        processed_docs = utils.lemmatize_texts(processed_docs)
        
        # Create dictionary and corpus
        self.dictionary = Dictionary(processed_docs)
        corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        return corpus
    
    def train(self, corpus, passes=20):
        """Train the LDA model."""
        self.model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            passes=passes
        )
        
    def get_document_topics(self, doc):
        """Get topic distribution for a single document."""
        processed_doc = utils.preprocess_text(doc)
        processed_doc = utils.remove_custom_stopwords([processed_doc])[0]
        processed_doc = utils.lemmatize_texts([processed_doc])[0]
        bow = self.dictionary.doc2bow(processed_doc)
        return self.model.get_document_topics(bow)
    
    def get_topic_terms(self, topic_id, num_words=10):
        """Get the most relevant terms for a given topic."""
        return self.model.show_topic(topic_id, num_words)
    
    def get_topic_distribution(self, docs):
        """Get topic distribution for multiple documents."""
        corpus = self.prepare_corpus(docs)
        return [self.model.get_document_topics(doc) for doc in corpus]
    
    def save_model(self, path):
        """Save the trained model."""
        self.model.save(path)
    
    def load_model(self, path):
        """Load a trained model."""
        self.model = LdaModel.load(path) 