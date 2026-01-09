from collections import defaultdict, Counter
import math
import re
from typing import Iterable, Dict


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
        self.num_docs_by_category = defaultdict(int)  # categoryCounts

        self.vocab = set()
        self.word_freq = defaultdict(Counter)

    def train(self, category, txt):
        self.num_docs += 1
        self.num_docs_by_category[category] += 1

        words = [w for w in tokenize(txt)]
        self.vocab.update(words)
        self.word_freq[category].update(words)

    def classify(self, txt, verbose=False) -> Dict[str, float]:
        if self.num_docs == 0:
            return {cat: 1.0 / len(self.categories) for cat in self.categories}

        words = [w for w in tokenize(txt)]
        V = max(len(self.vocab), 1)  # avoid division by zero

        # small optimization to avoid unnecessary calcs when words appears multiple times
        doc_counts = Counter(words)

        if verbose:
            print("#" * 20)
            print("TEXT:", txt)
            print("TOKENS:", doc_counts)
            print("#" * 20)

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

            if verbose:
                print("#" * 20)
                print(f"\nClass: {cat}")
                print(f"  Prior: {math.exp(log_prior):.4f}")

            for w, n in doc_counts.items():
                count = self.word_freq[cat][w]

                p = (count + self.alpha) / denom
                contrib = n * math.log(p)
                log_likelihood += contrib

                if verbose:
                    print(
                        f"  {w:<12} n={n:<2} count={count:<2} p={p:.6f} contrib={contrib:.4f}"
                    )

            log_scores[cat] = log_prior + log_likelihood

        # Convert log-scores to normalized probabilities (softmax)
        m = max(log_scores.values())
        exp_scores = {k: math.exp(v - m) for k, v in log_scores.items()}
        Z = sum(exp_scores.values()) or 1.0
        probs = {k: v / Z for k, v in exp_scores.items()}

        if verbose:
            print("#" * 20)
            for cat, p in probs.items():
                print(f"Final P({cat})={p:.4f}")
            print("#" * 20)

        return probs

    def stats(self, top_n=10):
        print("=== Classifier Stats ===")
        print(f"Total documents: {self.num_docs}")
        print(f"Vocabulary size: {len(self.vocab)}\n")

        print("Number of docs per category", dict(self.num_docs_by_category))

        for cat in self.categories:
            doc_count = self.num_docs_by_category[cat]
            prior = doc_count / self.num_docs if self.num_docs else 0.0
            token_count = sum(self.word_freq[cat].values())

            print(f"Category: {cat}")
            print(f"  Documents: {doc_count}")
            print(
                f"  Prior P({cat}): {prior:.3f}"
            )  # How likely a category is before you look at the text?
            print(f"  Total tokens: {token_count}")

            if token_count == 0:
                print("  (no tokens)")
            else:
                print(f"  Top {top_n} words:")
                for w, c in self.word_freq[cat].most_common(top_n):
                    print(f"    {w:<15} {c}")
            print()
