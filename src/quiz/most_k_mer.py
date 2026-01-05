#!/usr/bin/env python3

import sys

def most_abundant_k_mer(seq, k):

    dict_kmer = {}

    for i in range(0, len(seq)-k+1):
        subseq = seq[i:i+k]
        if subseq not in dict_kmer:
            dict_kmer[subseq] = 1
        else:
            dict_kmer[subseq] += 1

    max_value = max(dict_kmer.values())
    max_key = max(dict_kmer, key=dict_kmer.get)

    return max_key, max_value


def main() -> int:

    subseq, freq = most_abundant_k_mer("atggcgtaggcatg", 3)

    print(f'Most abundant k mer is "{subseq}", appearing {freq} time(s)')
    # Most abundant k mer is "atg", appearing 2 time(s)

    return 0


if __name__ == '__main__':
    sys.exit(main())

