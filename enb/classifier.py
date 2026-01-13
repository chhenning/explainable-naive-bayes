from collections import defaultdict, Counter
import math
import re
from typing import Iterable, Dict, Optional, Set

from enb.stats import Stats
from enb.porter import PorterStemmer

DEFAULT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def tokenize(
    doc: str,
    stop_words: Optional[Set[str]] = None,
    stemmer: Optional[PorterStemmer] = None,
):
    """
    Returns a list of a all words without getting tripped up by a single apostrophe.
    Also, ignore any punctuations.

    Example:
        - "Don't stop believing" -> ['don't', 'stop', 'believing']
        - "Hi! It's me." -> ['hi', "it's", 'me']
    """
    # simple, decent tokenizer
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", doc.lower())
    if stop_words:
        words = [w for w in words if w not in stop_words]

    if stemmer:
        words = [stemmer.stem_word(w) for w in words]

    return words


class Classifier:
    def __init__(
        self,
        categories: Iterable[str],
        alpha: float = 1.0,
        stop_words: Optional[Iterable[str]] = None,
        stem: bool = False,
    ):
        self.alpha = float(alpha)
        self.stop_words = (
            set(stop_words) if stop_words is not None else DEFAULT_STOP_WORDS
        )

        self.stemmer = PorterStemmer() if stem else None

        self.categories = list(categories)

        self.num_docs = 0
        self.num_docs_by_category = defaultdict(int)

        self.vocab = set()
        self.word_freq = defaultdict(Counter)

        self.stats = Stats()

    def train(self, category, doc: str):
        """
        Train the classifier with a text labeled with a category.

        :param category: The category label for this text.
        :param doc: The text to train on.
        """
        self.num_docs += 1
        self.num_docs_by_category[category] += 1

        words = [w for w in tokenize(doc, self.stop_words, self.stemmer)]
        self.vocab.update(words)
        self.word_freq[category].update(words)

    def classify(self, doc: str, keep_stats: bool = True) -> Dict[str, float]:
        """
        Classify a text and return the probabilities for each category.

        :param doc: The document to classify.
        :param keep_stats: If true, stores the stats for this classification.

        :return: Returns a dictionary mapping each category to its probability.
        :rtype: Dict[str, float]
        """
        # Handle the case where the classifier has not been trained yet
        if self.num_docs == 0:
            return {cat: 1.0 / len(self.categories) for cat in self.categories}

        if keep_stats:
            self.stats.set_num_docs(self.num_docs)
            self.stats.set_vocab(self.vocab)
            self.stats.set_num_docs_by_category(self.num_docs_by_category)

        words = [w for w in tokenize(doc, self.stop_words, self.stemmer)]
        V = max(len(self.vocab), 1)  # avoid division by zero

        # small optimization to avoid unnecessary calcs when words appears multiple times
        doc_counts = Counter(words)

        if keep_stats:
            self.stats.set_text_to_classify(doc, doc_counts)

        # multiplying probabilities is not good when values become really low. better to use log
        log_scores = {}

        for cat in self.categories:
            # Prior P(cat)
            cat_docs = self.num_docs_by_category[cat]

            if cat_docs == 0 or self.num_docs == 0:
                # unseen class (or untrained classifier)
                log_prior = float("-inf")
            else:
                log_prior = math.log(cat_docs / self.num_docs)

            # Likelihood P(words | cat)
            total_words_in_cat = sum(self.word_freq[cat].values())
            denom = total_words_in_cat + self.alpha * V

            log_likelihood = 0.0

            if keep_stats:
                self.stats.add_cat_prior(cat, log_prior)

            for w, n in doc_counts.items():
                count = self.word_freq[cat][w]

                # p is P(w | cat) with Laplace smoothing
                p = (count + self.alpha) / denom

                contrib = n * math.log(p)
                log_likelihood += contrib

                if keep_stats:
                    self.stats.add_word_contribution(cat, w, n, count, p, contrib)

            log_scores[cat] = log_prior + log_likelihood

        # Convert log-scores to normalized probabilities (softmax)
        m = max(log_scores.values())
        exp_scores = {k: math.exp(v - m) for k, v in log_scores.items()}
        Z = sum(exp_scores.values()) or 1.0
        probs = {k: v / Z for k, v in exp_scores.items()}

        if keep_stats:
            self.stats.add_results(probs)

        return probs

    def explain(self, doc) -> Stats:
        self.classify(doc, keep_stats=True)
        return self.stats
