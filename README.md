# explainable-naive-bayes

A simple, explainable Naive Bayes text classifier focused on transparency and first principles, written in plain Python.

## Why?

This project is built for **learning and transparency**.

Most modern machine learning models are "black boxes"—you see the result, but you don't know _why_ they made that choice. Naive Bayes is different. It is easy to look under the hood and see exactly which words pushed the classifier toward a specific category.

This implementation is written in plain Python from first principles. It shows how to build a stable and explainable classifier. If you need a high-performance, production-ready version, I recommend [scikit-learn's MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html).

### Features

**Log-Space Stability**
Probabilities are often tiny numbers that can break calculations when multiplied together. This implementation uses logs to chain calculations, ensuring the model stays stable and accurate even with very large documents or vocabularies.

**Explainable Statistics**
The system generates detailed reports, including confusion matrices to show exactly where the model is getting confused and word-level contributions that highlight which specific words drove a classification decision.

**Laplace Smoothing**
To handle words that the model hasn't seen during training, I use Laplace (additive) smoothing. This prevents "zero-probability" errors and allows the classifier to generalize better to new, unseen text.

**Data Preprocessing**
The pipeline includes built-in support for stop word removal and Porter stemming. These steps help reduce noise by ignoring common filler words and reducing different forms of a word (like "running" and "runs") to their root form.

**Multi-Class Classification**
Unlike simple binary classifiers, this implementation supports an unlimited number of categories. You can train it on any number of labels, from sentiment analysis to complex topic categorization.

**Pure Python & No Dependencies**
Built with first principles in mind, the core classifier uses only standard Python libraries. It is lightweight, easy to read, and can be integrated into any project without worrying about complex dependency chains.

**Command-Line Interface**
A built-in CLI makes it easy to experiment with different datasets. You can list available datasets, run full evaluations, or ask the model to explain its reasoning for a specific piece of text.

## Usage

### Install dependencies

This project uses the `uv` package manager.

```bash
git clone https://github.com/chhenning/explainable-naive-bayes.git
uv sync
```

### List available datasets

```bash
make ls
```

```sh
DATASET NAME     DESCRIPTION
------------     -----------
fake_newsgroup   Simple default test dataset
20_newsgroups    Scikit-Learn exported dataset (23MB)
```

The `20_newsgroups` is extracted from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html#)

### Run script

There are two use case.

1. `Train` and `Predict` a labeled dataset
2. `Train` and `Explain` one document

#### Train and Predict

```sh
make run
```

```sh
Running with dataset: fake_newsgroup
Accuracy: 1.0

Confusion Matrix:
true \ pred         comp.graphics  rec.sport.baseball  sci.space  talk.politics.guns
------------------  -------------  ------------------  ---------  ------------------
comp.graphics       1              0                   0          0
rec.sport.baseball  0              1                   0          0
sci.space           0              0                   1          0
talk.politics.guns  0              0                   0          1
Time taken: 0.00 seconds
```

#### Train and Explain

```bash
python enb/app.py run -ds "fake_newsgroup" --explain "Public opinion on firearm regulation varies widely by region, often influenced by cultural and historical factors."
```

```sh
Stats
-----
num_docs_trained  : 8
vocab_size        : 122
doc_tokens        : 12
doc_unique_tokens : 12
doc_preview       : Public opinion on firearm regulation varies widely by region, often influenced by cultural and historical factors.

Categories
----------
category            num_docs  log_prior  prior
------------------  --------  ---------  --------
comp.graphics       2         -1.386294  0.250000
rec.sport.baseball  2         -1.386294  0.250000
sci.space           2         -1.386294  0.250000
talk.politics.guns  2         -1.386294  0.250000

Final probabilities
-------------------
category            P(category|doc)
------------------  ---------------
talk.politics.guns  0.583695
sci.space           0.184494
rec.sport.baseball  0.145924
comp.graphics       0.085887

Top word contributions (top_n_words=10)
---------------------------------------

[comp.graphics]
(no contributing words found in this category)

[rec.sport.baseball]
(no contributing words found in this category)

[sci.space]
(no contributing words found in this category)

[talk.politics.guns]
word    count_in_doc  count_in_cat  P(word|cat)  log_contrib
------  ------------  ------------  -----------  -----------
public  1             1             0.01290323   -4.350278
often   1             1             0.01290323   -4.350278
Time taken: 0.00 seconds
```

## Next Steps

### Improve quality

- add bigrams
- limit vocab size

### Metrics

#### Precision / Recall / F1 (macro)

For multiclass, use macro F1 as a great “one number” metric.

- Precision (per class): of what you predicted as class X, how many were truly X?
- Recall (per class): of all true X, how many did you catch?
- F1: harmonic mean of precision and recall
- Macro F1: average F1 across classes (treats each class equally)

Why macro F1 is good: if one class is rare, accuracy can look good while you still fail that class.

#### Log loss (cross-entropy)

- Penalizes confident wrong predictions heavily.
- You care about probability quality (not just the final label).

#### Cross-validation

- Use a k-fold cross-validation and report average accuracy / macro F1

## Acknowledgements

The original idea for this model came from `The Coding Train` [Coding Challenge 187: Bayes Theorem](https://www.youtube.com/watch?v=g3-PXyF8U70).
