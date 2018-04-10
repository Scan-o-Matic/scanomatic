from __future__ import absolute_import
import os.path
import random


def get_generic_name():
    bird = random.choice(get_bird_list())
    adjective = get_adjective(bird)
    return "{} {}".format(adjective.capitalize(), bird)


def get_bird_list():
    filename = os.path.join(os.path.dirname(__file__), 'birds.txt')
    with open(filename, 'r') as namefile:
        return [line.strip() for line in namefile.readlines() if line.strip()]


def get_adjective_list():
    filename = os.path.join(os.path.dirname(__file__), 'adjectives.txt')
    with open(filename, 'r') as namefile:
        return [line.strip() for line in namefile.readlines() if line.strip()]


def get_adjective(bird):
    adjectives = get_adjective_list()
    starts = [word[0].lower() for word in bird.split(' ')]
    alliterations = [adj for adj in adjectives if adj[0] in starts]
    if alliterations:
        return random.choice(alliterations)
    return random.choice(adjectives)
