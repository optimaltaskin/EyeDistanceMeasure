from components.logger import init_logger
from components.helpers import filename_no_extension

import os
import argparse


def arguement_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_file", help="full path to the config file",
                        required=True)
    parser.add_argument("--check_file", help="full path to the checkpoint file",
                        required=True)
    parser.add_argument("--root_folder", help="path to the root folder where pth and config files are located. "
                                              "Inference results will be saved here.", required=True)
    return parser.parse_args()


test_set_folders = ["result_validation",
                    "different_cards1"]


def main():
    args = arguement_parse()
    init_logger()
    config = args.conf_file
    checkpoint = args.check_file
    root = args.root_folder

    for set in test_set_folders:
        os.system("python detect_mask_batch.py "
                  f"--conf_file {config} "
                  f"--check_file {checkpoint} "
                  "--print_mask True "
                  f"--images_dir data/{set}/ "
                  f"--save_to_folder {root}/{filename_no_extension(checkpoint)}/inference_results/{set}/")


if __name__ == "__main__":
    main()
