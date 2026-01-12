import unittest

from enb.classifier import Classifier, tokenize
from enb.porter import PorterStemmer

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

    def test_tokenizer_with_stemming(self):
        ps = PorterStemmer()
        r = tokenize("ponies are running", stemmer=ps)
        # ponies -> poni, are -> are (stop word if default used, but here we only test stemming), running -> run
        self.assertIn("poni", r)
        self.assertIn("run", r)

    def test_classifier_with_stemming(self):
        c = Classifier(["cat1"], stem=True)
        c.train("cat1", "ponies are running")
        self.assertIn("poni", c.vocab)
        self.assertIn("run", c.vocab)
        self.assertNotIn("ponies", c.vocab)
        self.assertNotIn("running", c.vocab)


    def test_training(self):
        c = Classifier(train_data.keys())

        for cat, txts in train_data.items():
            for txt in txts:
                c.train(cat, txt)

        self.assertTrue(len(c.categories) == 3)
        self.assertTrue(c.num_docs == 24)
        self.assertTrue(len(c.vocab) == 63)

        self.assertDictEqual(c.num_docs_by_category, {"pos": 8, "neu": 8, "neg": 8})

        total_tokens = {cat: sum(c.word_freq[cat].values()) for cat in c.categories}
        self.assertDictEqual(total_tokens, {"pos": 31, "neu": 21, "neg": 29})


class TestPorterStemmer(unittest.TestCase):
    def setUp(self):
        self.stemmer = PorterStemmer()

    def test_basic_stemming(self):
        self.assertEqual(self.stemmer.stem_word("caresses"), "caress")
        self.assertEqual(self.stemmer.stem_word("ponies"), "poni")
        self.assertEqual(self.stemmer.stem_word("ties"), "ti")
        self.assertEqual(self.stemmer.stem_word("caress"), "caress")
        self.assertEqual(self.stemmer.stem_word("cats"), "cat")

    def test_step1ab(self):
        self.assertEqual(self.stemmer.stem_word("feed"), "feed")
        self.assertEqual(self.stemmer.stem_word("agreed"), "agre")
        self.assertEqual(self.stemmer.stem_word("disabled"), "disabl")
        self.assertEqual(self.stemmer.stem_word("matting"), "mat")
        self.assertEqual(self.stemmer.stem_word("mating"), "mate")
        self.assertEqual(self.stemmer.stem_word("meeting"), "meet")
        self.assertEqual(self.stemmer.stem_word("milling"), "mill")
        self.assertEqual(self.stemmer.stem_word("messing"), "mess")

    def test_more_complex_stems(self):
        self.assertEqual(self.stemmer.stem_word("relational"), "relat")
        self.assertEqual(self.stemmer.stem_word("conditional"), "condit")
        self.assertEqual(self.stemmer.stem_word("rational"), "ration")
        self.assertEqual(self.stemmer.stem_word("valenci"), "valenc")
        self.assertEqual(self.stemmer.stem_word("hesitanci"), "hesit")
        self.assertEqual(self.stemmer.stem_word("digitizer"), "digit")
        self.assertEqual(self.stemmer.stem_word("conformabli"), "conform")
        self.assertEqual(self.stemmer.stem_word("radicalli"), "radic")
        self.assertEqual(self.stemmer.stem_word("differentli"), "differ")
        self.assertEqual(self.stemmer.stem_word("vileli"), "vile")
        self.assertEqual(self.stemmer.stem_word("analogousli"), "analog")
        self.assertEqual(self.stemmer.stem_word("vietnamization"), "vietnam")
        self.assertEqual(self.stemmer.stem_word("predication"), "predic")
        self.assertEqual(self.stemmer.stem_word("operator"), "oper")
        self.assertEqual(self.stemmer.stem_word("feudalism"), "feudal")
        self.assertEqual(self.stemmer.stem_word("decisiveness"), "decis")
        self.assertEqual(self.stemmer.stem_word("hopefulness"), "hope")
        self.assertEqual(self.stemmer.stem_word("callousness"), "callous")
        self.assertEqual(self.stemmer.stem_word("formaliti"), "formal")
        self.assertEqual(self.stemmer.stem_word("sensitiviti"), "sensit")
        self.assertEqual(self.stemmer.stem_word("sensibiliti"), "sensibl")

