from __future__ import absolute_import
import random


def get_generic_name():
    return "Generic {}".format(random.choice(get_name_list()))


def get_name_list(filename='scanomatic/util/birds.txt'):
    with open(filename, 'r') as namefile:
        return [line.strip() for line in namefile.readlines()]
