

import os
import shutil
import logging


def create_project_folder(path):

    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)

    os.makedirs(path)



def define_logging(path):

    f = os.path.join(path, "logging.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig( \
                        # filename=f,
                        # filemode='w',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler(f),
                            logging.StreamHandler()])


    # logger = logging.getLogger("logger")

    msg = f"Logging output: {f}"
    logging.info("")
    logging.info(msg)
