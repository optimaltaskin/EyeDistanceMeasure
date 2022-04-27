from components.helpers import filename_no_extension

import os
import argparse


def get_lr_and_wd(filename: str) -> (str, str):
    # sample filename: lr_0.0001_wd_5e-05.pth. Extract lr and wd values.
    fname_no_ext = filename_no_extension(filename)
    print(f"Filename without extension is: {fname_no_ext}")
    name_segments = fname_no_ext.split("_")
    return name_segments[1], name_segments[3]


def place_lr_wd_in_conf_file(lr: str, wd: str, conf_file_lines: []):
    for i, l in enumerate(conf_file_lines):
        if l.startswith("optimizer"):
            optimizer_line = l.replace("{LEARNING_RATE}", str(lr))
            optimizer_line = optimizer_line.replace("{WEIGHT_DECAY}", str(wd))
            conf_file_lines[i] = optimizer_line
            print(f"Config file changed successfully!..")
            return


parser = argparse.ArgumentParser()

parser.add_argument("--pth_folder", help="Folder where all pth files are located(scans all sub folders as well)",
                    required=True)
parser.add_argument("--val_images", help="Folder location where all test images are located",
                    required=True)
parser.add_argument("--conf_template", help="Folder location where all test images are located",
                    required=False, default="conf_template.py")

args = parser.parse_args()
pth_folder = args.pth_folder
val_images = args.val_images

if not os.path.isdir(pth_folder) or not os.path.isdir(val_images):
    print(f"One of the input folders are not found! Folders are:\n{pth_folder}\n{val_images}")
    exit()

with open(os.path.join(pth_folder, args.conf_template)) as conf_file:
    conf_file_lines = conf_file.readlines()

for root, dirs, files in os.walk(pth_folder):
    new_conf = conf_file_lines.copy()

    for file in files:
        if '.pth' in file:
            lr, wd = get_lr_and_wd(file)
            pth_file = os.path.join(root, file)
            place_lr_wd_in_conf_file(lr, wd, new_conf)
            print(f"File: {pth_file}")
            print(f"lr: {lr}   - wd: {wd}")

            new_conf_file = os.path.join(root, f"config_pointrend50_lr_{lr}_wd_{wd}.py")
            print(f"Saving new config file to: {new_conf_file}")
            with open(new_conf_file, 'w+') as target_conf_file:
                target_conf_file.writelines(new_conf)
                print("Config file saved successfully")
            new_results_folder = os.path.join(root, "results/")
            os.makedirs(new_results_folder, exist_ok=True)
            os.system(f"python detect_mask_batch.py "
                      f"--conf_file={new_conf_file} "
                      f"--check_file='{pth_file}' "
                      f"--print_mask=True "
                      f"--save_to_folder={new_results_folder} ")

