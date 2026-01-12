from collections import defaultdict, Counter
import math
import re
from typing import Iterable, Dict

from enb.stats import Stats


def tokenize(txt: str):
    """
    Returns a list of a all words without getting tripped up by a single apostrophe.
    Also, ignore any punctuations.

    Example:
        - "Don't stop believing" -> ['don't', 'stop', 'believing']
        - "Hi! It's me." -> ['hi', "it's", 'me']
    """
    # simple, decent tokenizer
    return re.findall(r"[a-z]+(?:'[a-z]+)?", txt.lower())


class Classifier:
    def __init__(self, categories: Iterable[str], alpha: float = 1.0):
        self.alpha = float(alpha)

        self.categories = list(categories)

        self.num_docs = 0
        self.num_docs_by_category = defaultdict(int)

        self.vocab = set()
        self.word_freq = defaultdict(Counter)

        self.stats = Stats()

    def train(self, category, txt):
        """
        Train the classifier with a text labeled with a category.

        :param category: The category label for this text.
        :param txt: The text to train on.
        """
        self.num_docs += 1
        self.num_docs_by_category[category] += 1

        words = [w for w in tokenize(txt)]
        self.vocab.update(words)
        self.word_freq[category].update(words)

    def classify(self, txt, keep_stats=True) -> Dict[str, float]:
        """
        Classify a text and return the probabilities for each category.

        :param txt: The text to classify.
        :param verbose: If true, prints detailed calculation steps.

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

        words = [w for w in tokenize(txt)]
        V = max(len(self.vocab), 1)  # avoid division by zero

        # small optimization to avoid unnecessary calcs when words appears multiple times
        doc_counts = Counter(words)

        if keep_stats:
            self.stats.set_text_to_classify(txt, doc_counts)

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
