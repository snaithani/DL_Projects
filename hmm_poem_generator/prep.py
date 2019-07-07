import string
import collections

pi = collections.defaultdict(int)
second_given_first = collections.defaultdict(lambda:
                                             collections.defaultdict(int))
transitions = collections.defaultdict(lambda: collections.defaultdict(int))


def add_to_pi(key):
    pi[key] += 1


def add_to_dict(dic, key, val):
    dic[key][val] += 1


def normalize_pi():
    n = sum(pi.values())
    for k in pi:
        pi[k] /= n


def normalize_dict(dic):
    for k in dic:
        curr = dic[k]
        n = sum(curr.values())
        for v in dic[k]:
            curr[v] /= n


def get_dist():
    for line in open('data/robert_frost.txt'):
        tokens = remove_punctuations(line.rstrip().lower()).split()
        for i, token in enumerate(tokens):
            if i == 0:
                add_to_pi(token)
            else:
                if i == 1:
                    add_to_dict(second_given_first, tokens[0], tokens[1])
                else:
                    add_to_dict(transitions, (tokens[i - 2], tokens[i - 1]),
                                tokens[i])
                    if i == len(tokens) - 1:
                        add_to_dict(transitions, (tokens[i - 1], tokens[i]),
                                    'END')

    normalize_pi()
    normalize_dict(second_given_first)
    normalize_dict(transitions)

    return pi, second_given_first, transitions


def remove_punctuations(token):
    return token.translate(str.maketrans('', '', string.punctuation))


if __name__ == '__main__':
    get_dist()
