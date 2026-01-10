import argparse
from pprint import pprint

from enb import Classifier, create_dataset_from_json, accuracy


def main():
    train, test = create_dataset_from_json("data/fake_newsgroup.json")

    c = Classifier(set(t["label"] for t in train))
    for t in train:
        c.train(t["label"], t["text"])

    results = []
    for t in test:
        probs = c.classify(t["text"])
        predicted = max(probs, key=probs.get)
        results.append((t["label"], predicted))

    print("Accuracy:", accuracy(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text Classification using Naive Bayes."
    )

    commands_subparser = parser.add_subparsers(dest="command")
    show_subparser = commands_subparser.add_parser(name="ls")

    # run_subparser = parser.add_subparsers(dest="run")
    # run_subparser.add_argument("-ds", "--dataset", type=str, required=True)

    kwargs = vars(parser.parse_args())
    command = kwargs.pop("command")

    if command == "ls":
        print(f"{'DATASET NAME':<15}  DESCRIPTION")
        print(f"{'------------':<15}  -----------")
        print(f"{'fake_newsgroup':<15}  Simple default test dataset")
        print(f"{'20_newsgroupsE':<15}  Sklearn exported dataset (23MB)")

    # main()
