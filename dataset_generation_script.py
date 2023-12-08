import os

# Paths to your directories
gt_dirs = {
    'AFW': 'data/AFW_GT',
    #'IBUG': 'data/IBUG_GT',
    #'LFPW': 'data/LFPW_GT',
    #'HELEN': 'data/HELEN_GT',
}
output_file = 'train_data_file.txt'

def find_gt_file(base_name, dataset_name):
    """Finds the corresponding ground truth file for a given input file."""
    gt_dir = gt_dirs[dataset_name]
    if dataset_name in ['AFW', 'HELEN']:
        gt_path = os.path.join(gt_dir, base_name + '.npz')
    else:
        gt_path = os.path.join(gt_dir, base_name + '.npy')
    return gt_path

def process_dataset(dataset_name):
    """Processes each dataset and writes to the output file."""
    # dataset_dir = os.path.join(input_dir, dataset_name)
    with open(output_file, 'a') as file:
        for input_file in os.listdir(gt_dirs[dataset_name]):
            if input_file.endswith('_facymap.jpg'):
                base_name = input_file.rsplit('_facymap.jpg', 1)[0]
                input_path = os.path.join(gt_dirs[dataset_name], base_name + '.jpg')
                gt_path = find_gt_file(base_name, dataset_name)
                file.write(f'{input_path} {gt_path}\n')

# Process each dataset
for dataset_name in gt_dirs.keys():
    process_dataset(dataset_name)
