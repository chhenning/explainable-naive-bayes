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
    main()
