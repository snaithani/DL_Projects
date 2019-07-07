from prep import get_dist
import numpy as np


def get_first_word(pi):
    rand = np.random.rand()
    so_far = 0
    word = None

    for k in pi:
        so_far += pi[k]
        if rand < so_far:
            word = k
            break

    return word


def get_next_word(dic, prevKey):
    curr = dic[prevKey]

    rand = np.random.rand()
    so_far = 0
    word = None

    for k in curr:
        so_far += curr[k]
        if rand < so_far:
            word = k
            break

    return word


def get_sequence(pi, second_given_first, transitions):
    first_word = get_first_word(pi)
    second_word = get_next_word(second_given_first, first_word)

    sentence = first_word + ' ' + second_word

    i = 2

    while True:
        word = get_next_word(transitions, (first_word, second_word))
        if word == 'END':
            break
        sentence += ' ' + word
        first_word = second_word
        second_word = word
        i += 1

    return sentence


def poem(pi, second_given_first, transitions, num_lines=4):
    poem = ''

    for i in range(num_lines):
        poem += get_sequence(pi, second_given_first, transitions) + '\n'

    return poem


if __name__ == '__main__':
    pi, second_given_first, transitions = get_dist()
    print(poem(pi, second_given_first, transitions, 6))
