import os
import shutil

conf_file_no_ext = "point_rend_50_tuning"
file_path = f"configs/_optimal/{conf_file_no_ext}.py"
target_conf_file_path = f"configs/_optimal/{conf_file_no_ext}_final.py"
tuning_res_folder = "results/tuning_results/"

lr_list = [0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
weight_decay_list = [0.0001, 0.00005, 0.00001, 0.000005]

target_conf_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), target_conf_file_path)

with open(file_path) as conf_file:
    conf_file_lines = conf_file.readlines()

for lr in lr_list:
    for wd in weight_decay_list:
        is_conf_changed: bool = False
        for i, l in enumerate(conf_file_lines):
            if l.startswith("optimizer"):
                optimizer_line = l.replace("{LEARNING_RATE}", str(lr))
                optimizer_line = optimizer_line.replace("{WEIGHT_DECAY}", str(wd))
                conf_file_lines[i] = optimizer_line
                print(f"Config file changed successfully!..")
                is_conf_changed = True
                break
        if not is_conf_changed:
            print("CONFIG FILE NOT CHANGED!! LEARNING RATE AND WEIGHT DECAY ARE NOT MODIFIED!")

        with open(target_conf_file_path, 'w+') as target_conf_file:
            target_conf_file.writelines(conf_file_lines)

        os.system(f"python tools/train.py {target_conf_file_path}")
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                f"{tuning_res_folder}lr_{lr}_wd_{wd}")
        print(f"created directory: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
        os.system(f"python detect_mask_batch.py "
                  f"--conf_file={target_conf_file_path} "
                  f"--check_file='work_dirs/{conf_file_no_ext}_final/latest.pth' "
                  f"--single_step=True "
                  f"--image_file=data/result_validation/26.jpg "
                  f"--print_mask=True "
                  f"--save_to_folder={tuning_res_folder}lr_{lr}_wd_{wd}/")
        from_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 f"work_dirs/{conf_file_no_ext}_final/epoch_64.pth")
        to_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               f"{tuning_res_folder}lr_{lr}_wd_{wd}/lr_{lr}_wd_{wd}.pth")
        shutil.move(from_file, to_file)
