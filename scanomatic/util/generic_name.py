from random import Random


def get_generic_name(seed):
    rand = Random(seed)
    return "Generic {}".format(rand.choice(get_name_list()))


def get_name_list(filename='scanomatic/util/birds.txt'):
    with open(filename, 'r') as namefile:
        return [line.strip() for line in namefile.readlines()]
