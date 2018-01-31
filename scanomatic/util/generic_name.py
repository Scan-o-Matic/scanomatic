from __future__ import absolute_import
import os.path
import random


def get_generic_name():
    return "Generic {}".format(random.choice(get_name_list()))


def get_name_list():
    filename = os.path.join(os.path.dirname(__file__), 'birds.txt')
    with open(filename, 'r') as namefile:
        return [line.strip() for line in namefile.readlines()]
