





def remove_white_space_at_ends(text, start, end):

    # leading white space
    n = len(text)
    text = text.lstrip()
    start += n - len(text)

    # trailing white space
    n = len(text)
    text = text.rstrip()
    end -= n - len(text)

    return (text, start, end)
