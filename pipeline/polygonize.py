import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.features
import shapely.geometry
import shapely.ops
import shapely.wkt
import torch
from matplotlib.collections import PatchCollection
from skimage import io
from tqdm import tqdm

from models.unet.unet import Unet
from models.unet.unet_baseline import UnetBaseline

import polygonize_config


logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logging.info('Using PyTorch version %s.', torch.__version__)


save_pred_polygons = polygonize_config.CONFIG['save_pred_polygons']

# Polygonization parameters
min_polygon_area = polygonize_config.CONFIG['min_polygon_area']
use_buffer = polygonize_config.CONFIG['use_buffer']
buffer_size = polygonize_config.CONFIG['buffer_size']

model_choice = 'unetbase'  #'unetbase' or 'unetv2'
feature_scale = 1  # 2 is the same as unetbase

use_gpu = False
dtype = torch.float32


if use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
logging.info('Using device: %s.', device)


def visualize_poly(poly_list, mask, out_path):
    """
    Visualizes the polygons produced by mask_to_poly() and save them at the specified path

    Args:
        poly_list: list of shapely.geometry.polygon.Polygon on this image
        mask: the predicted mask, needed for laying out the axes
        out_path: path at which the visualization of the list of polygons is to be saved
    """
    fig, ax = plt.subplots()
    ax.imshow(mask, alpha=0)  # don't show the mask, but need this to be added to the axes for polygons to show up
    patch_list = []

    for poly in poly_list:
        x, y = poly.exterior.coords.xy
        xy = np.column_stack((x, y))
        polygon = matplotlib.patches.Polygon(xy, linewidth=1, edgecolor='b', facecolor='none')
        patch_list.append(polygon)

    p = PatchCollection(patch_list, cmap=matplotlib.cm.jet, alpha=1)
    ax.add_collection(p)

    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def mask_to_poly(mask, image_id, count_border_as_background=True):
    """
    Convert from 256x256 mask to polygons on the 256x256 image
    Adapted from original code: https://github.com/SpaceNetChallenge/BuildingDetectors_Round2/tree/master/1-XD_XD
    Args:
        mask: a numpy array of shape (256, 256, 4) from io.imread(test_path)
        image_id: unique part of the image file name
        count_border_as_background: whether to assign border pixels as background

    Returns:
        df: a dataframe listing the required fields for each polygon, needed for SpaceNet utilities to compute the F-1 score.
        polygons: a list of shapely.geometry.polygon.Polygon, which are the polygons on this image
    """

    # only need to sum across color channels if mask is read from a saved image
    # mask = np.sum(mask, axis=2) # make grey scale

    # for 'jet' colormap
    # if count_border_as_background:
    #     mask[mask == 775] = 408  # as background
    # else:
    #     mask[mask == 775] = 571  # as building

    if count_border_as_background:  # 0 background, 1 building, 2 border
        mask[mask == 2] = 0  # as background
    else:
        mask[mask == 2] = 1  # as building

    # this function uses a default of 4 pixel connectivity for grouping pixels into features
    shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)

    polygons = []
    for shape, val in shapes:
        s = shapely.geometry.shape(shape).exterior

        if use_buffer:
            s = shapely.geometry.polygon.Polygon(s.buffer(buffer_size))
        else:
            s = shapely.geometry.polygon.Polygon(s)

        if s.area > min_polygon_area:
            polygons.append(s)

    mp = shapely.geometry.MultiPolygon(polygons)

    if isinstance(mp, shapely.geometry.Polygon):
        df = pd.DataFrame({
            'area_size': [mp.area],
            'poly': [mp],
            'image_id': [image_id]
        })
    else:
        df = pd.DataFrame({
            'area_size': [p.area for p in mp],
            'poly': [p for p in mp],
            'image_id': [image_id] * len(mp)
        })

    df = df.sort_values(by='area_size', ascending=False)
    df.loc[:, 'wkt'] = df.poly.apply(lambda x: shapely.wkt.dumps(x, rounding_precision=0))
    df.loc[:, 'bid'] = list(range(1, len(df) + 1))
    df.loc[:, 'area_ratio'] = df.area_size / df.area_size.max()
    return df, polygons


def main(cp_path, input_image_dir, out_path, vis_dir=None, save_pred=save_pred_polygons):
    """
    Applies the model at cp_path to input images and output the csv required for SpaceNet to
    compute the F-1 score and other metrics against the ground truth.

    Args:
        cp_path: path to the model checkpoint to use
        input_image_dir: path to directory containing the images to extract building footprints from,
         usually the val or test dir
        out_path: path of the output csv
        vis_dir: optionally a directory to place the visualization of polygons on each image
        save_pred: whether to save visualizations to vis_dir
    """
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    checkpoint = torch.load(cp_path)

    if model_choice == 'unetv2':
        model = Unet(feature_scale=feature_scale, n_classes=3, is_deconv=True, in_channels=3, is_batchnorm=True)
    elif model_choice == 'unetbase':
        model = UnetBaseline(feature_scale=feature_scale, n_classes=3, is_deconv=True, in_channels=3, is_batchnorm=True)
    else:
        raise ValueError('Unknown model_choice={0}'.format(model_choice))

    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device=device, dtype=dtype)
    model.eval()  # set model to evaluation mode
    logging.info('Model loaded from checkpoint.')

    result_dfs = []

    image_files = os.listdir(input_image_dir)
    image_files = [image_file for image_file in image_files if image_file.endswith('.jpg')]

    for image_name in tqdm(image_files):
        image_name_no_file_type = image_name.split('.jpg')[0]

        image_id = image_name_no_file_type.split('RGB-PanSharpen_')[1]  # of format _-115.3064538_36.1756826998
        image_path = os.path.join(input_image_dir, image_name)
        original_image = io.imread(image_path)

        image = original_image.transpose((2, 0, 1))
        image = torch.from_numpy(np.expand_dims(image, 0)).type(torch.float32).to(device=device, dtype=dtype)

        with torch.no_grad():
            scores = model(image)
            _, prediction = scores.max(1)

        prediction = prediction.reshape((256, 256)).cpu().data.numpy()

        result_df, polygons = mask_to_poly(prediction, image_id)
        result_dfs.append(result_df)

        # save prediction polygons visualization to output
        if save_pred and vis_dir:
            visualize_poly(polygons, prediction, os.path.join(vis_dir, 'poly_' + image_name))

    all_df = pd.concat(result_dfs)

    logging.info('Writing result to csv, length of all_df is {}'.format(len(all_df)))
    with open(out_path, 'w') as f:
        f.write('ImageId,BuildingId,PolygonWKT_Pix,Confidence\n')

        for i, row in tqdm(all_df.iterrows()):
            f.write("{},{},\"{}\",{:.6f}\n".format(
                row.image_id,
                int(row.bid),
                row.wkt,
                row.area_ratio))


if __name__ == '__main__':
    cp_path = polygonize_config.CONFIG['cp_path'],
    input_image_dir = polygonize_config.CONFIG['input_image_dir'],
    out_path = polygonize_config.CONFIG['out_path'],
    vis_dir = polygonize_config.CONFIG['vis_dir'] if polygonize_config.CONFIG['vis_dir'] else None

    main(cp_path, input_image_dir, out_path, vis_dir=vis_dir)

