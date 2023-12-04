import os

# Paths to your directories
input_dir = '300W_LP_INPUT'
gt_dirs = {
    'AFW': 'AFW_GT',
    'IBUG': 'IBUG_GT',
    'LFPW': 'LFPW_GT'
}
output_file = 'train_data_file.txt'

def find_gt_file(input_file, dataset_name):
    """Finds the corresponding ground truth file for a given input file."""
    base_name = os.path.splitext(input_file)[0]
    gt_dir = gt_dirs[dataset_name]
    if dataset_name in ['AFW', 'HELEN']:
        gt_path = os.path.join(gt_dir, base_name + '.npz')
    else:
        gt_path = os.path.join(gt_dir, base_name + '.npy')
    return gt_path

def process_dataset(dataset_name):
    """Processes each dataset and writes to the output file."""
    dataset_dir = os.path.join(input_dir, dataset_name)
    with open(output_file, 'a') as file:
        for input_file in os.listdir(dataset_dir):
            if input_file.endswith('.jpg'):
                input_path = os.path.join(dataset_dir, input_file)
                gt_path = find_gt_file(input_file, dataset_name)
                file.write(f'{input_path} {gt_path}\n')

# Process each dataset
for dataset_name in gt_dirs.keys():
    process_dataset(dataset_name)
