from Orange import data, classification


class Learn:
    def __init__(self):
        self.data = data.Table("estilo")
        _nb = classification.NaiveBayesLearner()
        classifier = _nb(data)
        target_class = 1
        print("Probabilities for %s:" % data.domain.class_var.values[target_class])
        probabilities = classifier(data, 1)
        for p, d in zip(probabilities[5:8], data[5:8]):
            print(p[target_class], d.get_class())


if __name__ == '__main__':
    Learn()
