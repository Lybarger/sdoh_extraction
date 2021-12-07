
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer

def get_length_percentiles(text, transformer, path=None, description=None):

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(transformer)

    # tokenize text
    encodings = tokenizer(text, \
                padding = "max_length",
                return_tensors = 'pt')
    mask = encodings['attention_mask'].numpy()

    # get document lengths
    lengths = np.sum(mask, axis=1)

    # get percentiles
    x = range(0, 101)
    percentiles = np.percentile(lengths, x)

    # packet percentile as dataframe
    df = pd.DataFrame(zip(x, percentiles), columns=["percentile", "length"])

    if path is not None:
        if description is None:
            f = os.path.join(path, "length_percentile.csv")
        else:
            f = os.path.join(path, f"length_percentile_{description}.csv")
        df.to_csv(f, index=False)

    return df
