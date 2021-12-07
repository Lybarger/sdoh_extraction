

import os
from pathlib import Path
import logging
import glob
import re



def get_sacred_config_file(dir, pattern='config.json'):


    # find all config files
    files = list(Path(dir).rglob(pattern))

    # make sure at least one configfile found
    assert len(files) > 0

    # make sure file comes from sacred directory
    files = [file for file in files if re.match('[0-9]+', file.parent.name)]

    # make sure at least one configfile found
    assert len(files) > 0

    # get most recent
    file = max(files, key=os.path.getctime)

    return file
