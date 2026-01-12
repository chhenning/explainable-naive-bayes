from collections import defaultdict, Counter
from typing import Dict, Optional, Any
import math


from collections import defaultdict, Counter

# def stats(self, top_n=10):
#     # print("=== Classifier Stats ===")
#     # print(f"Total documents: {self.num_docs}")
#     # print(f"Vocabulary size: {len(self.vocab)}\n")

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

    # ---------------- Pretty printing helpers ----------------

    @staticmethod
    def _fmt_float(x: Optional[float], digits: int = 6) -> str:
        if x is None:
            return "—"
        try:
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return str(x)
            return f"{x:.{digits}f}"
        except Exception:
            return str(x)

    @staticmethod
    def _fmt_int(x: Optional[int]) -> str:
        if x is None:
            return "—"
        try:
            return f"{int(x):,}"
        except Exception:
            return str(x)

    @staticmethod
    def _truncate(s: Optional[str], max_len: int = 140) -> str:
        if not s:
            return ""
        s = " ".join(s.split())  # collapse whitespace
        return s if len(s) <= max_len else s[: max_len - 1] + "…"

    @staticmethod
    def _print_section(title: str) -> None:
        print()
        print(title)
        print("-" * len(title))

    @staticmethod
    def _print_kv(rows: list[tuple[str, str]]) -> None:
        if not rows:
            return
        key_w = max(len(k) for k, _ in rows)
        for k, v in rows:
            print(f"{k:<{key_w}} : {v}")

    @staticmethod
    def _print_table(headers: list[str], rows: list[list[Any]]) -> None:
        if not headers:
            return
        # Convert all cells to strings
        srows = [[("" if c is None else str(c)) for c in r] for r in rows]
        widths = [len(h) for h in headers]
        for r in srows:
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(c))

        def fmt_row(r: list[str]) -> str:
            return "  ".join(f"{c:<{widths[i]}}" for i, c in enumerate(r))

        print(fmt_row(headers))
        print("  ".join("-" * w for w in widths))
        for r in srows:
            print(fmt_row(r))

    # ---------------- Main printer ----------------

    def print_to_output(self, top_n_words: int = 10) -> None:
        # General
        self._print_section("Stats")
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

        self._print_kv(
            [
                ("num_docs_trained", self._fmt_int(self.num_docs)),
                ("vocab_size", self._fmt_int(vocab_size)),
                ("doc_tokens", self._fmt_int(doc_token_count)),
                ("doc_unique_tokens", self._fmt_int(doc_unique)),
                ("doc_preview", self._truncate(self.doc, 200)),
            ]
        )

        # Docs by category + priors
        if self.num_docs_by_category or self.cat_priors:
            self._print_section("Categories")
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
                        self._fmt_int(n),
                        self._fmt_float(lp, 6),
                        self._fmt_float(math.exp(lp) if lp is not None else None, 6),
                    ]
                )
            self._print_table(["category", "num_docs", "log_prior", "prior"], cat_rows)

        # Final probabilities
        if self.probs:
            self._print_section("Final probabilities")
            # sort descending by prob
            items = sorted(self.probs.items(), key=lambda kv: kv[1], reverse=True)
            prob_rows = [[c, self._fmt_float(p, 6)] for c, p in items]
            self._print_table(["category", "P(category|doc)"], prob_rows)

        # Word contributions per category
        if self.word_contributions:
            self._print_section(f"Top word contributions (top_n_words={top_n_words})")

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
                            self._fmt_int(d.get("word_count")),
                            self._fmt_int(d.get("word_count_in_cat")),
                            self._fmt_float(d.get("P(word|category)"), 8),
                            self._fmt_float(d.get("log_contrib"), 6),
                        ]
                    )

                self._print_table(
                    [
                        "word",
                        "count_in_doc",
                        "count_in_cat",
                        "P(word|cat)",
                        "log_contrib",
                    ],
                    rows,
                )
