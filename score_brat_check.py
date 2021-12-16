

import argparse
import os
import sys

import config.constants as C

from score_brat import get_argparser

def main(args):


    print(args)


if __name__ == '__main__':

    parser = get_argparser()
    args = parser.parse_args()

    main(args)
