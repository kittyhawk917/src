#!/usr/bin/env python3

import sys

def prime_numbers(max):
    prime_nums = []

    for i in range(2, max+1):
        j = 2
        is_prime = True
        while j < i:
            mod = i % j
            if i % j == 0:
                is_prime = False
                break
            j = j + 1
        
        if is_prime:
            prime_nums.append(i)

    return prime_nums


def main() -> int:
    
    print(prime_numbers(100))
    # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    return 0


if __name__ == '__main__':
    sys.exit(main())

