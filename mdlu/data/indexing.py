import json
import os
import tempfile
from functools import partial

import yaml
from jsonargparse import CLI
from loguru import logger
from pydicom import dcmread
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm


def items_in_dir(directory: str):
    subdirs, files = [], []
    for x in os.listdir(directory):
        full_x = os.path.join(directory, x)
        if os.path.isdir(full_x):
            subdirs.append(full_x)
        elif os.path.isfile(full_x):
            files.append(full_x)
        else:
            raise RuntimeError(f"{full_x} is neither a file nor a directory!")
    return sorted(subdirs), sorted(files)


def find_leaf_dirs(directory):
    logger.debug(f"Find leafdirs in {directory}")
    dirs_to_process = [directory]
    leaf_dirs = []
    while dirs_to_process:
        curr_dir = dirs_to_process.pop(0)

        subdirs, files = items_in_dir(curr_dir)
        if subdirs:
            dirs_to_process.extend(subdirs)
        else:
            leaf_dirs.append(curr_dir)

    logger.debug(f"Sort {len(leaf_dirs)} leafdirs")
    return sorted(leaf_dirs)


def query_relevant_information_single_dir(leaf_dir: str, query_keys: list):
    meta_data = {}
    for f in items_in_dir(leaf_dir)[1]:
        try:
            dcm = dcmread(f)
            for k in query_keys:
                meta_data[k] = getattr(dcm, k)
            break
        except:
            continue

    return meta_data


def process_whole_dir_tree(
    root_dir: str,
    query_keys: list,
    store_temp_leafdirs: bool = True,
    num_workers: int = 10,
):
    leaf_dirs = None
    if store_temp_leafdirs:
        temp_path = tempfile.gettempdir()
        temp_store_file = os.path.join(temp_path, "leafdirs_" + os.path.split(root_dir)[1] + ".json")

        if os.path.isfile(temp_store_file):
            with open(temp_store_file) as f:
                leaf_dirs = json.load(f)

    if leaf_dirs is None:
        leaf_dirs = find_leaf_dirs(root_dir)

    if store_temp_leafdirs:
        if not os.path.isfile(temp_store_file):
            with open(temp_store_file, "w") as f:
                json.dump(leaf_dirs, f)

    func = partial(query_relevant_information_single_dir, query_keys=query_keys)

    if num_workers > 0:
        meta_data = process_map(
            func,
            leaf_dirs,
            max_workers=num_workers,
            desc="Retrieve MetaData from dirs",
        )
    else:
        meta_data = map(func, tqdm(leaf_dirs, desc="Retrieve MetaData from dirs"))
    final_meta_data = {}
    could_not_process = []
    for d, m in zip(leaf_dirs, meta_data):
        if m:
            final_meta_data[str(d).replace(str(root_dir), "")] = m
        else:
            could_not_process.append(d)

    return final_meta_data, could_not_process


def recursive_query_information(
    root_dir: str, output_path: str, query_keys: list, store_temp_leafdirs: bool, num_workers: int = 10
):
    final_meta_data, could_not_process = process_whole_dir_tree(
        root_dir=root_dir, query_keys=query_keys, store_temp_leafdirs=store_temp_leafdirs, num_workers=num_workers
    )
    message_files = "\n\t- ".join(could_not_process)

    logger.info("Could not process the following directories (they don't contain dicom files):\n\t" + message_files)

    with open(output_path, "w") as f:
        if output_path.endswith(".yaml") or output_path.endswith(".yml"):
            yaml.dump(final_meta_data, f)
        elif output_path.endswith(".json"):
            json.dump(final_meta_data, f)
        else:
            raise ValueError(
                f"Could not estimate filetype from file-extension. Supported are JSON (.json) and YAML (.yaml or .yml) but got {output_path}"
            )


def _main():
    CLI(recursive_query_information, as_positional=False)


if __name__ == "__main__":
    _main()
