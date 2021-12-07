import os
import shutil
import logging


def make_and_clear(dest, recursive=False, clear_=True):
    '''
    Create dest if it does not exist and
    remove any files, but not folders, in dest if exists

    NOTE: All of this printing is not necessary. It is here for some troubleshooting with Condor.
    '''

    # Create destination folder
    if os.path.exists(dest) and clear_:
        if recursive:
            #logging.info("Recursively removing:\t{}".format(dest))
            shutil.rmtree(dest)
            #logging.info("Recursive removal complete")

        else:

            # Remove files in dest (not recursive)
            files_to_remove = [os.path.join(dest, f) for f in os.listdir(dest) \
              if os.path.isfile(os.path.join(dest, f))]

            #logging.info("Non-recursively removing {} files from:\t{}".format(len(files_to_remove), dest))
            for f in files_to_remove:
                os.remove(f)


    if not os.path.exists(dest):
        #logging.info("Creating directory:\t{}".format(dest))
        os.makedirs(dest)

    return True

def setup_dir(dest, root=None, scratch=None, clear_scratch=True):

    '''
    Project folder
    '''
    #print("\nProject folder set up")
    _ = make_and_clear(dest, recursive=False, clear_=True)

    '''
    Scratch space
    '''

    if scratch is not None:
        # Set up scratch
        root_d = os.path.basename(root)
        relpath = os.path.relpath(dest, start=root)
        scratch_ = os.path.join(scratch, root_d, relpath)
        #print("\nScratch folder set up")
        _ = make_and_clear(scratch_, recursive=True, clear_=clear_scratch)


        return scratch_
    else:
        return None
