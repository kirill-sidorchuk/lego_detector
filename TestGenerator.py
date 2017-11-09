import argparse

from DataGenerator import DataGenerator


def test_generator(args):
    generator = DataGenerator(args.data_root, "train", 90, (0.7, 1.2), (1, 3), 0, 0, 0, 32, (224, 224), 3, False, False, False)

    n_samples = 100
    i = 0
    for img in generator.generate():
        i += 1
        if i > n_samples:
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Finetune model')
    parser.add_argument("data_root", type=str, help="data root dir")

    _args = parser.parse_args()
    test_generator(_args)