import random


def get_generic_name(seed):
    random.seed(seed)
    return "Generic {}".format(random.sample(ANIMALS, 1)[0])


ANIMALS = [
    "Gamefowl",
    "Galliform",
    "Gazelle",
    "Gecko",
    "Gerbil",
    "Giant panda",
    "Giant squid",
    "Gibbon",
    "Gila monster",
    "Giraffe",
    "Goat",
    "Golden eagle",
    "Goldfish",
    "Goose",
    "Gopher",
    "Gorilla",
    "Grasshopper",
    "Great blue heron",
    "Great white shark",
    "Grebe",
    "Grizzly bear",
    "Ground shark",
    "Ground sloth",
    "Grouse",
    "Guan",
    "Guanaco",
    "Guineafowl",
    "Guinea pig",
    "Gull",
    "Guppy",
]
