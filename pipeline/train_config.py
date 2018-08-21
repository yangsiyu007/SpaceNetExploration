import torch


# for explanation also see comments in train.py, top part of file

TRAIN = {
    # hardware and framework parameters
    'use_gpu': True,
    'dtype': torch.float32,
    'cudnn_benchmark': True,

    # paths to data splits
    'data_path_root': '~/building_extraction/sample_data/', # common part of the path for data_path_train, data_path_val and data_path_test
    'data_path_train': 'Vegas_8bit_256_train',
    'data_path_val': 'Vegas_8bit_256_val',
    'data_path_test': 'Vegas_8bit_256_test',

    # training and model parameters
    'model_choice': 'unet_baseline',  # 'unet_baseline' or 'unet'
    'feature_scale': 1,  # parameter for the Unet

    'num_workers': 4,  # how many subprocesses to use for data loading
    'train_batch_size': 10,
    'val_batch_size': 4,
    'test_batch_size': 4,

    'starting_checkpoint_path': '',  # checkpoint .tar to train from, empty if training from scratch
    'loss_weights': [0.1, 0.8, 0.1],  # weight given to loss for pixels of background, building interior and building border classes
    'learning_rate': 0.5e-3,
    'print_every': 50,  # print every how many steps
    'total_epochs': 15,  # for the walkthrough, we are training for one epoch

    'experiment_name': 'unet_interior_weights', # using weights that emphasize the building interior pixels
}


