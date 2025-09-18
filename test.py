

import json
import os
import datasets


def dataset_to_json(dataset, filename, ):
    data_nums = len(dataset)
    with open(filename, "w") as f:
        for i in range(data_nums):
            row_data = dataset[i]
            json_data = json.dumps(row_data)
            f.write(json_data)
            f.write('\n')

def get_task_data(data_path, split, task_id, unlearned_tasks):
    forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split + '.json'), split='train')
    forget_pertrubed_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split + '_perturbed.json'),
                                                  split='train')

    retain_split = "retain" + str(100 - min(10 * int(split.replace("forget", "")), 90)).zfill(2)
    retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split + '.json'),
                                        split='train')


    forget_retain_data = forget_data.filter(lambda x: int(x['task_id']) not in unlearned_tasks)
    curr_forget_data = forget_data.filter(lambda x: int(x['task_id']) == task_id)

    curr_retain_data = datasets.concatenate_datasets([retain_data, forget_retain_data])

    curr_forget_perturbed_data = forget_pertrubed_data.filter(lambda x: int(x['task_id']) == task_id)

    return curr_forget_data, curr_retain_data



    # 随机选择50%的curr_forget_data数据
    if len(curr_forget_data) > 0:
        original_size = len(curr_forget_data)
        # 打乱数据顺序
        curr_forget_data = curr_forget_data.shuffle(seed=42)
        # 选择前50%的数据
        half_size = len(curr_forget_data) // 2
        curr_forget_data = curr_forget_data.select(range(half_size))
        if local_rank == 0:
            print(f"任务 {cfg.task_id}: 原始forget数据量: {original_size}, 随机选择50%后数据量: {len(curr_forget_data)}")

