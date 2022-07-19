


import os



def get_include(params):


    include = []

    for p in params:
        if p is not None:
            include.append(p)

    if len(include) == 0:
        return None
    else:
        return set(include)
