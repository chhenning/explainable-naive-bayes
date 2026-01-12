from collections import defaultdict, Counter
from typing import Dict
import math

from enb.printing_helpers import (
    _fmt_float,
    _fmt_int,
    _print_kv,
    _print_section,
    _print_table,
    _truncate,
)

# def stats(self, top_n=10):
#     # print("Number of docs per category", dict(self.num_docs_by_category))

#     for cat in self.categories:
#         doc_count = self.num_docs_by_category[cat]
#         prior = doc_count / self.num_docs if self.num_docs else 0.0
#         token_count = sum(self.word_freq[cat].values())

#         print(f"Category: {cat}")
#         print(f"  Documents: {doc_count}")
#         print(
#             f"  Prior P({cat}): {prior:.3f}"
#         )  # How likely a category is before you look at the text?
#         print(f"  Total tokens: {token_count}")

#         if token_count == 0:
#             print("  (no tokens)")
#         else:
#             print(f"  Top {top_n} words:")
#             for w, c in self.word_freq[cat].most_common(top_n):
#                 print(f"    {w:<15} {c}")
#         print()


class Stats:
    """
    Classifier statistics for a single classification.
    """

    def __init__(self):

        # the total number of documents the classifier has been trained on
        self.num_docs = 0

        # the vocabulary used by the classifier
        self.vocab = None

        # number of documents by category
        self.num_docs_by_category = defaultdict(int)

        # P(cat) - prior probabilities for each category
        self.cat_priors = {}

        # word contributions for each category
        self.word_contributions = defaultdict(list)

        # the text being classified
        self.doc = None

        # word counts in the text being classified
        self.doc_vocab = None

        # final probabilities for each category
        self.probs = {}

    def set_num_docs(self, n):
        self.num_docs = n

    def set_vocab(self, vocab):
        self.vocab = vocab

    def set_text_to_classify(self, doc: str, doc_vocab: Counter):
        self.doc = doc
        self.doc_vocab = doc_vocab

    def set_num_docs_by_category(self, num_docs_by_category: Dict[str, int]):
        self.num_docs_by_category = dict(num_docs_by_category)

    def add_cat_prior(self, category: str, log_prior: float):
        self.cat_priors[category] = log_prior

    def add_word_contribution(
        self,
        category: str,
        word: str,
        word_count: int,
        word_count_in_cat: int,
        prob: float,
        log_contrib: float,
    ):
        """
        Adds a word contribution entry for a given category.

        :param category: Category label.
        :type category: str
        :param word: The word being analyzed.
        :type word: str
        :param word_count: The count of the word in the text.
        :type word_count: int
        :param word_count_in_cat: The count of the word in the category.
        :type word_count_in_cat: int
        :param prob: The probability P(word|category).
        :type prob: float
        :param log_contrib: The log contribution of the word to the category score.
        :type log_contrib: float
        """
        self.word_contributions[category].append(
            {
                "word": word,
                "word_count": word_count,
                "word_count_in_cat": word_count_in_cat,
                "P(word|category)": prob,
                "log_contrib": log_contrib,
            }
        )

    def add_results(self, probs: Dict[str, float]):
        self.probs = probs

    def print_to_output(self, top_n_words: int = 10) -> None:
        """
        Prints the stats to the output.

        :param top_n_words: Number of words to show in the word contributions.
        """
        # General
        _print_section("Stats")
        vocab_size = None
        if self.vocab is not None:
            try:
                vocab_size = len(self.vocab)
            except Exception:
                vocab_size = None

        doc_vocab = getattr(self, "doc_vocab", None)
        doc_token_count = None
        doc_unique = None
        if isinstance(doc_vocab, Counter):
            doc_token_count = sum(doc_vocab.values())
            doc_unique = len(doc_vocab)

        _print_kv(
            [
                ("num_docs_trained", _fmt_int(self.num_docs)),
                ("vocab_size", _fmt_int(vocab_size)),
                ("doc_tokens", _fmt_int(doc_token_count)),
                ("doc_unique_tokens", _fmt_int(doc_unique)),
                ("doc_preview", _truncate(self.doc, 200)),
            ]
        )

        # Docs by category + priors
        if self.num_docs_by_category or self.cat_priors:
            _print_section("Categories")
            cats = sorted(
                set(self.num_docs_by_category.keys()) | set(self.cat_priors.keys())
            )
            cat_rows = []
            for c in cats:
                n = self.num_docs_by_category.get(c)
                lp = self.cat_priors.get(c)
                cat_rows.append(
                    [
                        c,
                        _fmt_int(n),
                        _fmt_float(lp, 6),
                        _fmt_float(math.exp(lp) if lp is not None else None, 6),
                    ]
                )
            _print_table(["category", "num_docs", "log_prior", "prior"], cat_rows)

        # Final probabilities
        if self.probs:
            _print_section("Final probabilities")
            # sort descending by prob
            items = sorted(self.probs.items(), key=lambda kv: kv[1], reverse=True)
            prob_rows = [[c, _fmt_float(p, 6)] for c, p in items]
            _print_table(["category", "P(category|doc)"], prob_rows)

        # Word contributions per category
        if self.word_contributions:
            _print_section(f"Top word contributions (top_n_words={top_n_words})")

            for category in sorted(self.word_contributions.keys()):
                contribs = self.word_contributions.get(category, [])
                if not contribs:
                    continue

                # EXCLUDE words not seen in the category
                contribs = [d for d in contribs if d.get("word_count_in_cat", 0) > 0]

                if not contribs:
                    print()
                    print(f"[{category}]")
                    print("(no contributing words found in this category)")
                    continue

                # sort by absolute log contribution (largest impact first)
                sorted_contribs = sorted(
                    contribs,
                    key=lambda d: abs(d.get("log_contrib", 0.0) or 0.0),
                    reverse=True,
                )[: max(0, int(top_n_words))]

                print()
                print(f"[{category}]")

                rows = []
                for d in sorted_contribs:
                    rows.append(
                        [
                            d.get("word", ""),
                            _fmt_int(d.get("word_count")),
                            _fmt_int(d.get("word_count_in_cat")),
                            _fmt_float(d.get("P(word|category)"), 8),
                            _fmt_float(d.get("log_contrib"), 6),
                        ]
                    )

                _print_table(
                    [
                        "word",
                        "count_in_doc",
                        "count_in_cat",
                        "P(word|cat)",
                        "log_contrib",
                    ],
                    rows,
                )
