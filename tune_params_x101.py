from components.logger import init_logger

import os
import shutil
import glob
import json
import logging

init_logger("tune_params.log")
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("PIL").setLevel(logging.INFO)

conf_file_no_ext = "conf_template_m7X101"
file_path = f"configs/_optimal/{conf_file_no_ext}.py"
target_conf_file_path = f"configs/_optimal/{conf_file_no_ext}_final.py"
tuning_res_folder = "results/tuning_results/AdamW/mask7/"

NUMBER_OF_EPOCHS = 50
EXCLUDE_ITER_JSON = "exclude_iters.json"
exclude_iter_json_path = f'{tuning_res_folder}{EXCLUDE_ITER_JSON}'

bbone_list = [101]
set_list = ["cocoset_10k_american_cards"]
mask_list = [7]
lr_list = [0.0001, 0.00005, 0.00001]

target_conf_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), target_conf_file_path)

with open(file_path) as conf_file:
    conf_file_lines = conf_file.readlines()

if os.path.exists(exclude_iter_json_path):
    with open(exclude_iter_json_path, 'r') as exc_iter_file:
        excluded_iterations = json.load(exc_iter_file)
else:
    excluded_iterations = []

logger.info(f"Starting with excluded iterations as below: \n{excluded_iterations}")
for bbone in bbone_list:
    for set in set_list:
        for m in mask_list:
            for lr in lr_list:
                for wd in [lr, lr * 0.1]:
                    logger.info(f"Beginning iteration with following parameters:\nmask_size:{m}\nset:{set}\nlr:{lr}\nwd:{wd}")

                    # check if those parameters are in excluded list
                    skip_iteration: bool = False
                    for exc in excluded_iterations:
                        if set == exc["set"] and m == exc["mask_size"] and lr == exc["lr"] and wd == exc["wd"] and \
                                exc["bbone"] == bbone:
                            skip_iteration = True

                    if skip_iteration:
                        logger.info(f"Skipping iteration with following parameters:\nmask_size:{m}\nset:{set}\nlr:{lr}\nwd:{wd}")
                        continue

                    for i, l in enumerate(conf_file_lines):
                        if l.strip().startswith("optimizer"):
                            optimizer_line = l.replace("{LEARNING_RATE}", str(lr))
                            optimizer_line = optimizer_line.replace("{WEIGHT_DECAY}", str(wd))
                            conf_file_lines[i] = optimizer_line
                            logger.info(f"Learning rate and weight decay changed successfully!..")
                        elif l.strip().startswith("img_prefix") or l.strip().startswith("ann_file"):
                            folder_line = l.replace("{SET_FOLDER}", str(set))
                            conf_file_lines[i] = folder_line
                            logger.info(f"Set folder changed successfully!..")
                        elif l.strip().startswith("roi_layer"):
                            roi_line = l.replace("{OUTPUT_SIZE}", str(int(m / 2)))
                            conf_file_lines[i] = roi_line
                            logger.info(f"Output size changed successfully!..")
                        elif l.strip().startswith("mask_size"):
                            mask_line = l.replace("{MASK_SIZE}", str(m))
                            conf_file_lines[i] = mask_line
                            logger.info(f"Mask size changed successfully!..")
                        elif l.strip().startswith("checkpoint_config") or l.strip().startswith("runner"):
                            epoch_line = l.replace("{EPOCH}", str(NUMBER_OF_EPOCHS))
                            conf_file_lines[i] = epoch_line
                            logger.info(f"Number of epochs changed successfully!..")
                        elif l.strip().startswith("depth"):
                            bbone_line = l.replace("{BBONE}", str(bbone))
                            conf_file_lines[i] = bbone_line
                            logger.info(f"Backbone depth changed successfully!..")
                        elif l.strip().startswith("checkpoint"):
                            if bbone == 50:
                                bbone_pretrain = "'open-mmlab://detectron2/resnet50_caffe'"
                            elif bbone == 101:
                                bbone_pretrain = "'open-mmlab://detectron2/resnet101_caffe'"
                            else:
                                logger.info("Backbone selection had an undetermined issue!.. Exiting app...")
                                exit()
                            bbone_line = l.replace("{BBONE_PRETRAIN}", str(bbone_pretrain))
                            conf_file_lines[i] = bbone_line
                            logger.info(f"Backbone pretrain selection changed successfully!..")

                    with open(target_conf_file_path, 'w+') as target_conf_file:
                        target_conf_file.writelines(conf_file_lines)
                    logger.info("Config file saved successfully. Beginning training!")
                    pth_filename = f"set_{set}_bbone{bbone}_m_{m}_lr_{lr}_wd_{wd}"
                    log_filename = f"{pth_filename}_log_file"
                    try:
                        os.system(f"python tools/train.py {target_conf_file_path} "
                                  f"--log_name={log_filename}")
                    except Exception as e:
                        logger.info(f"Training returned an exception. Error message: {e}")
                        continue

                    target_folder_name = f"{tuning_res_folder}{pth_filename}"
                    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), target_folder_name)

                    logger.info(f"Training completed. Created directory: {dir_path}.")
                    os.makedirs(dir_path, exist_ok=True)
                    try:
                        os.system(f"python run_test.py "
                                  f"--conf_file={target_conf_file_path} "
                                  f"--check_file='work_dirs/{conf_file_no_ext}_final/latest.pth' "
                                  f"--root_folder={target_folder_name}/")
                    except Exception as e:
                        logger.info(f"Test run returned an exception. Error message: {e}")
                        continue
                    logger.info("Creating necessary folders to store resulting path and config files.")
                    from_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             f"work_dirs/{conf_file_no_ext}_final/epoch_{NUMBER_OF_EPOCHS}.pth")
                    to_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           f"{target_folder_name}/pth_{pth_filename}.pth")
                    shutil.move(from_file, to_file)

                    config_from_file = target_conf_file_path
                    config_to_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           f"{target_folder_name}/config_{pth_filename}.py")
                    shutil.move(config_from_file, config_to_file)

                    log_from_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 f"work_dirs/{conf_file_no_ext}_final/{log_filename}.log")
                    log_to_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               f"{target_folder_name}/{log_filename}.log.json")
                    shutil.move(log_from_file, log_to_file)

                    json_logs = glob.glob(f"work_dirs/{conf_file_no_ext}_final/*.json")
                    latest_json = max(json_logs, key=os.path.getctime)

                    json_log_from_file = os.path.realpath(latest_json)
                    json_log_to_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               f"{target_folder_name}/{log_filename}.log.json")
                    shutil.move(json_log_from_file, json_log_to_file)

                    plot_out = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            f"{target_folder_name}/loss_graphs.jpg")
                    logger.info("Plotting loss values...")
                    os.system(f"python tools/analysis_tools/analyze_logs.py "
                              f"plot_curve {log_to_file} "
                              f"--keys loss_rpn_bbox loss_mask loss_point loss "
                              f"--legend segm_rpn_mAP loss_mask loss_point loss "
                              f"--out={plot_out}")

                    excluded_iterations.append({"mask_size": m, "set": set, "lr": lr, "wd": wd, "bbone": bbone})

                    with open(exclude_iter_json_path, 'w') as fp:
                        json.dump(excluded_iterations, fp)
