

import logging
import random



def random_sample(X, size, seed=1, exclude=None, sort_values=True):


    logging.info("Random sampling")

    # make sure input is sorted list
    X = list(X)
    if sort_values:
        X = sorted(X)


    if exclude is not None:

        assert isinstance(exclude, (list,set))

        n0 = len(X)
        logging.info('Exclude len:     {}'.format(len(exclude)))
        logging.info('count, original: {}'.format(n0))
        exclude = set(exclude)

        X = [x for x in X if x not in exclude]

        n1 = len(X)
        logging.info('count, retained: {}'.format(n1))
        logging.info('count, removed:  {}'.format(n0 - n1))


    # Requested size greater than available documents
    if size >= len(X):
        logging.warn("Random sample: size > len ({} > {})".format(size, len(X)))
        Y = X
        size = len(Y)

    # Identify samples
    else:

        logging.info(f"Random seed fixed: {seed}")

        random.Random(seed).shuffle(X)
        Y = X[:size]


        assert len(Y) == len(set(Y))
        assert len(Y) == size

    if sort_values:
        Y = sorted(Y)

    return Y
