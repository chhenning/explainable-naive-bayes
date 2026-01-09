import unittest

from enb.classifier import Classifier, tokenize

train_data = {
    "pos": [
        "I love this product",
        "Amazing experience, super happy",
        "Fantastic service and great quality",
        "This made my day",
        "So good, I would buy again",
        "Everything was perfect",
        "Excellent, wow that was awesome",
        "I am very satisfied",
    ],
    "neu": [
        "It arrived on Tuesday",
        "The package was delivered",
        "I used it once",
        "It works as expected",
        "The size is medium",
        "The color is blue",
        "I received the email confirmation",
        "Setup took five minutes",
    ],
    "neg": [
        "I hate this product",
        "Terrible experience, very disappointed",
        "This is the worst purchase",
        "It broke after one day",
        "Awful service and poor quality",
        "Nothing works and I am frustrated",
        "I want a refund",
        "Completely useless",
    ],
}

test_samples = [
    ("pos", "Really great quality, I am happy with it"),
    ("pos", "Awesome service, excellent experience"),
    ("pos", "Love it, works perfectly"),
    ("neu", "It was delivered today"),
    ("neu", "The setup was quick and straightforward"),
    ("neu", "I used it twice this week"),
    ("neg", "Very disappointed, it stopped working"),
    ("neg", "Poor quality and terrible service"),
    ("neg", "Worst experience, I want my money back"),
]


class TestClassifier(unittest.TestCase):

    def test_tokenizer(self):
        r = tokenize("Don't stop believing")
        self.assertListEqual(r, ["don't", "stop", "believing"])

    def test_training(self):
        c = Classifier(train_data.keys())

        for cat, txts in train_data.items():
            for txt in txts:
                c.train(cat, txt)

        self.assertTrue(len(c.categories) == 3)
        self.assertTrue(c.num_docs == 24)
        self.assertTrue(len(c.vocab) == 72)

        self.assertDictEqual(c.num_docs_by_category, {"pos": 8, "neu": 8, "neg": 8})

        total_tokens = {cat: sum(c.word_freq[cat].values()) for cat in c.categories}
        self.assertDictEqual(total_tokens, {"pos": 35, "neu": 33, "neg": 35})
