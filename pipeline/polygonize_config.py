import torch

CONFIG = {
    # Paths

    # Path to the model checkpoint file to use for inference
    'cp_path': '../checkpoints/unet_checkpoint_epoch19_2018-06-03-04-00-35.pth.tar',

    # Path to the directory containing the validation images and annotations
    'input_image_dir': '~/building_extraction/sample_data/Vegas_8bit_256_val/annotations',

    # OPTIONAL Path to directory to save polygon proposal visualizations
    'vis_dir': '~/building_extraction/out/Vegas_8bit_256_val_poly/predictions',
    'save_pred_polygons': False,  # set to true to save polygon proposal visualizations

    # Path to csv to save the proposals in a csv
    'out_path': '~/building_extraction/out/Vegas_8bit_256_val_poly/proposals.csv',

    # Parameters for creating polygons from segmentation result
    'min_polygon_area': 150,

    'use_buffer': False,  # see Shapely documentation
    'buffer_size': 3.0,

    # model parameters - set to be the same as used in train_config.py
    'model_choice': 'unet_baseline',  # 'unet_baseline' or 'unet'
    'feature_scale': 1,  # parameter for the Unet

    # hardware and framework parameters
    'use_gpu': True,
    'dtype': torch.float32
}

