
import os


def filename_no_extension(path: str):
    basename = os.path.basename(path)
    filename = os.path.splitext(basename)

    return filename[0]
