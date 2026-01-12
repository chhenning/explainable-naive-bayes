# explainable-naive-bayes

A simple, explainable Naive Bayes text classifier focused on transparency and first principles, written in plain Python.


# Features

- chaining probability calculations using logs
- detailed stats output
- Laplace smoothing 


# Ideas

## Improve quality

- porter stemmer
- exclude stop words
- add bigrams
- limit vocab size

## Metrics

1) Accuracy

What it tells you: percent of texts classified correctly.
When it’s good: classes are roughly balanced and mistakes are equally bad.

2) Confusion matrix

What it tells you: which classes you confuse with which (e.g., neu → neg).
This is the most informative thing to look at first for a classifier like yours.

3) Precision / Recall / F1 (macro)

For multiclass, use macro F1 as a great “one number” metric.
	•	Precision (per class): of what you predicted as class X, how many were truly X?
	•	Recall (per class): of all true X, how many did you catch?
	•	F1: harmonic mean of precision and recall
	•	Macro F1: average F1 across classes (treats each class equally)

Why macro F1 is good: if one class is rare, accuracy can look good while you still fail that class.

Metrics for “explainable” probability outputs

Your model returns probabilities. So you can also measure how good those probabilities are:

4) Log loss (cross-entropy)

What it tells you: penalizes confident wrong predictions heavily.
When it’s good: you care about probability quality (not just the final label).

5) Brier score (optional)

Measures squared error of predicted probabilities. Often used for calibration-ish evaluation.

Practical advice for your repo

If you add just three things, do:
	1.	Confusion matrix
	2.	Macro F1
	3.	Accuracy

And optionally (nice touch for NB):
4. Log loss

Big caveat with your current dataset

Your sample data is tiny, so any single train/test split will be noisy. A better evaluation approach is:
	•	k-fold cross-validation (even 5-fold) and report average accuracy / macro F1
	•	or leave-one-out if the dataset is extremely small

If you want, I can drop in a small evaluate(test_samples) helper that computes accuracy + confusion matrix + macro F1 (no sklearn), and optionally log loss using the probabilities your classifier already outputs.

# Acknowledgements

The original idea for this model came from `The Coding Train` [Coding Challenge 187: Bayes Theorem](https://www.youtube.com/watch?v=g3-PXyF8U70).