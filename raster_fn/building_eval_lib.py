import logging
import numpy as np
import torch
import sys
sys.path.append('/repos/SpaceNetExploration')
from training.models.unet.unet import Unet
from training.models.unet.unet_baseline import UnetBaseline


"""
Performs inference on a tile of image of arbitrary size.

Functions here reference https://github.com/Azure/pixel_level_land_classification/blob/master/geoaidsvm/04_Apply_trained_model_in_ArcGIS_Pro.ipynb
"""


use_gpu = True
dtype = torch.float32

if use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
logging.info('Using device: %s.', device)


def get_cropped_data(image, bounds):
    a, b, c, d = bounds
    return np.asarray(image[:, int(a) : int((a + c)), int(b) : int(b + d)], dtype=np.float32)


def enumerate_squares(image, batch_size, block_size, padding):
    c, w, h = image.shape

    features = np.ones((int(batch_size), c, block_size, block_size), dtype=np.float32)
    coords = list(range(int(batch_size)))

    x_coords = [min((block_size - 2 * padding) * i, w - block_size) for i in
                range(int((w - padding) / (block_size - 2 * padding)))]
    y_coords = [min((block_size - 2 * padding) * i, h - block_size) for i in
                range(int((h - padding) / (block_size - 2 * padding)))]

    c_squares = 0
    for i in x_coords:
        for j in y_coords:
            bounds = (i, j, block_size, block_size)
            features[c_squares, :, :, :] = get_cropped_data(image, bounds)
            coords[c_squares] = (i, j)
            c_squares += 1
            if c_squares >= batch_size:
                yield c_squares, coords, features
                c_squares = 0

    if c_squares > 0:
        features = features[0:c_squares, :, :, :]
        yield c_squares, coords, features


def load_model(model_path, model_choice='unetbase', device=device):
    checkpoint = torch.load(model_path, map_location=device)

    feature_scale = 1  # Hardcoded right now
    if model_choice == 'unetv2':
        model = Unet(feature_scale=feature_scale, n_classes=3, is_deconv=True, in_channels=3, is_batchnorm=True)
    elif model_choice == 'unetbase':
        model = UnetBaseline(feature_scale=feature_scale, n_classes=3, is_deconv=True, in_channels=3, is_batchnorm=True)
    else:
        raise ValueError('Unknown model_choice={0}'.format(model_choice))

    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device=device, dtype=dtype)
    model.eval()  # set model to evaluation mode
    return model


def get_train_info(model):
    # TODO make non-hardcoded and read from model
    input_channels, classes, padding = 3, 3, 0
    return input_channels, classes, padding


def classify_tile(model, tile_data, output_data, batch_size=8):
    '''
tile_data is a numpy array of bytes of shape (c_input_channels, w, h)
output_data is a numpy array of bytes of shape (w, h)
    '''
    block_size = 256  # TODO make non-hardcoded
    input_channels, classes, padding = get_train_info(model)

    for size, coords, features in enumerate_squares(tile_data, batch_size, block_size, padding):
        # features are of size batch_size, channels, block_size, block_size
        img_input = torch.from_numpy(features).type(torch.float32).to(device=device, dtype=dtype)

        with torch.no_grad():
            scores = model(img_input)
            _, prediction = scores.max(1)

        scores = scores.cpu().data.numpy()

        for n in range(size):
            x, y = coords[n]
            temp = scores[n]
            output_data[:, int(x+padding):int(x+block_size-padding), int(y+padding):int(y+block_size-padding)] = temp[:, :, :]
    return prediction


colour_map = np.asarray([[0.7,0.7,1],[0.8, 1, 0.8],[0.9,0.6,0.1]], dtype=np.float32)
def render(softmax, hard):
    sum = np.sum(np.exp(softmax), axis=0)
    for i in range(softmax.shape[0]):
        softmax[i, :, :] = np.exp(softmax[i, :, :]) / sum
    cf, wf, hf = softmax.shape
    if hard:
        hardmax_pic = np.zeros((3, wf, hf))
        hardmax = softmax.argmax(axis=0)
        for c in range(cf):
            for ch in range(3):
                hardmax_pic[ch,:,:] += (hardmax == c) * colour_map[c, ch]
        return hardmax_pic
    else:
        softmax_pic = np.zeros((3, wf, hf))
        for c in range(cf):
            for ch in range(3):
                softmax_pic[ch, :, :] += softmax[c, :, :] * colour_map[c, ch]
        return softmax_pic

