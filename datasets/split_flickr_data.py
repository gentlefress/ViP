import random
import os

DATA_PATH = '../flickr30k/results_20130124.token'

train_ratio = 0.7
dev_ratio = 0.3


def shuffle_data(x):
    subscript = list(range(len(x)))
    random.Random(42).shuffle(subscript)
    shuffle_x = []
    for i in subscript:
        shuffle_x.append(x[i])
    return shuffle_x


def split_data(lines, train_len, dev_len):
    train = []
    dev = []
    for i in range(len(lines)):
        if len(train) < train_len:
            train.append(lines[i])
        else:
            if len(dev) < dev_len:
                dev.append(lines[i])
            else:
                break
    return train, dev


def generate_sen2img(lines):
    sen2img = {}
    for line in lines:
        data = line.strip().split('\t')
        img, sen = data[0].strip(), data[1].strip()
        sen2img[sen] = img
    return sen2img


def generate_samples(lines, sen2img):
    pos_samples = []
    for line in lines:
        data = line.strip().split('\t')
        img, sen = data[0].strip(), data[1].strip()
        pos = img + '\t' + sen + '\tmatched'
        pos_samples.append(pos)
        print(pos)
    return pos_samples


def write_data(lines, filename):
    if os.path.isdir(os.path.dirname(filename)) is False:
        os.mkdir(os.path.dirname(filename))
    sen2img = generate_sen2img(lines)
    positive_samples = generate_samples(lines, sen2img)
    samples = []
    for i in range(len(positive_samples)):
        samples.append(positive_samples[i])
    with open(filename, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(sample+'\n')


if __name__ == "__main__":
    original_lines = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        original_lines += lines

    # original_lines = shuffle_data(original_lines)
    original_train, original_dev = split_data(original_lines, int(len(original_lines)*train_ratio),
                                              len(original_lines) - int(len(original_lines)*train_ratio))
    write_data(original_train, '../flickr/train.tsv')
    write_data(original_dev, '../flickr/dev.tsv')
