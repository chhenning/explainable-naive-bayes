import argparse
from pprint import pprint
import time

from enb import Classifier, create_dataset_from_json, accuracy

datasets = {
    "fake_newsgroup": {
        "desc": "Simple default test dataset",
        "path": "data/fake_newsgroup.json",
    },
    "20_newsgroups": {
        "desc": "Scikit-Learn exported dataset (23MB)",
        "path": "data/20_newsgroups.json",
    },
}


def main(dataset: str):

    start_time = time.time()

    print(f"Running with dataset: {dataset}")

    train, test = create_dataset_from_json(datasets[dataset]["path"])

    c = Classifier(set(t["label"] for t in train))
    for t in train:
        c.train(t["label"], t["text"])

    results = []
    for t in test:
        probs = c.classify(t["text"], keep_stats=False)
        predicted = max(probs, key=probs.get)
        results.append((t["label"], predicted))

    print("Accuracy:", accuracy(results))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    ## Example of getting detailed probabilities for a single text

    probs, stats = c.explain(test[0]["text"])
    print("\nDetailed probabilities for first test document:")
    pprint(probs)
    pprint("\nStats:")
    pprint(stats.to_dict(top_n_words=5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text Classification using Naive Bayes."
    )

    commands_subparser = parser.add_subparsers(dest="command")
    ls_subparser = commands_subparser.add_parser(
        name="ls", description="List available datasets"
    )

    run_subparser = commands_subparser.add_parser(
        name="run", description="Run classification on a dataset"
    )
    run_subparser.add_argument(
        "-ds",
        "--dataset",
        dest="dataset",
        type=str,
        default="fake_newsgroup",
    )

    kwargs = vars(parser.parse_args())
    command = kwargs.pop("command")

    if command == "ls":
        print(f"{'DATASET NAME':<15}  DESCRIPTION")
        print(f"{'------------':<15}  -----------")
        print(f"{'fake_newsgroup':<15}  Simple default test dataset")
        print(f"{'20_newsgroups':<15}  Scikit-Learn exported dataset (23MB)")
    elif command == "run":
        ds = kwargs.pop("dataset")
        if ds not in datasets:
            raise ValueError(f"Unknown dataset: {ds}")
        main(ds)
    else:
        raise ValueError(f"Unknown command: {command}")
