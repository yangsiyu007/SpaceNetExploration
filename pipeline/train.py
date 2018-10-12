import logging
import os
import sys
from time import localtime, strftime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from models.unet.unet import Unet
from models.unet.unet_baseline import UnetBaseline

from utils.data_transforms import ToTensor
from utils.dataset import SpaceNetDataset
from utils.logger import Logger

import train_config


# train.py
# Execute from the root directory SpaceNetExploration to save checkpoints and logs there.


logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logging.info('Using PyTorch version %s.', torch.__version__)


# config for the training run

use_gpu = train_config.TRAIN['use_gpu']
dtype = train_config.TRAIN['dtype']
torch.backends.cudnn.benchmark = True  # enables benchmark mode in cudnn, see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936

data_path_root = train_config.TRAIN['data_path_root']
data_path_train = os.path.join(data_path_root, train_config.TRAIN['data_path_train'])
data_path_val = os.path.join(data_path_root, train_config.TRAIN['data_path_val'])
data_path_test = os.path.join(data_path_root, train_config.TRAIN['data_path_test'])

model_choice = train_config.TRAIN['model_choice']
feature_scale = train_config.TRAIN['feature_scale']

num_workers = train_config.TRAIN['num_workers']  # how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process
train_batch_size = train_config.TRAIN['train_batch_size']
val_batch_size = train_config.TRAIN['val_batch_size']
test_batch_size = train_config.TRAIN['test_batch_size']

# checkpoint to used to initialize, empty if training from scratch
# starting_checkpoint_path = './checkpoints/unet_checkpoint_epoch9_2018-05-26-21-52-44.pth.tar'
starting_checkpoint_path = ''

# weights for computing the loss function; absolute values of the weights do not matter
# [background, interior of building, border of building]
loss_weights = train_config.TRAIN['loss_weights']
score_weights_tensor = torch.from_numpy(np.array(loss_weights))
learning_rate = train_config.TRAIN['learning_rate']
print_every = train_config.TRAIN['print_every']
total_epochs = train_config.TRAIN['total_epochs']

experiment_name = train_config.TRAIN['experiment_name']

split_tags = ['trainval', 'test']  # compatibility tag, should always stay like this


# device configuration
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
logging.info('Using device: %s.', device)


# data sets and loaders
dset_train = SpaceNetDataset(data_path_train, split_tags, transform=T.Compose([ToTensor()]))
loader_train = DataLoader(dset_train, batch_size=train_batch_size, shuffle=True,
                          num_workers=num_workers) # shuffle True to reshuffle at every epoch

dset_val = SpaceNetDataset(data_path_val, split_tags, transform=T.Compose([ToTensor()]))
loader_val = DataLoader(dset_val, batch_size=val_batch_size, shuffle=True,
                        num_workers=num_workers) # also reshuffle val set because loss is recorded for the last batch

dset_test = SpaceNetDataset(data_path_test, split_tags, transform=T.Compose([ToTensor()]))
loader_test = DataLoader(dset_test, batch_size=test_batch_size, shuffle=True,
                         num_workers=num_workers)

logging.info('Training set size: {}, validation set size: {}, test set size: {}'.format(
    len(dset_train), len(dset_val), len(dset_test)))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()


def train_model(model, optimizer, epochs=1, print_every=10, checkpoint_path=''):
    """
    Train a model using the PyTorch Module API.

    Args:
    model: a PyTorch Module specifying the model to train.
    optimizer: an Optimizer object we will use to train the model
    epochs: optionally a Python int giving the number of epochs to train for

    Returns: logs model accuracies during training.
    """
    model = model.to(device=device, dtype=dtype)  # move the model parameters to CPU/GPU

    score_weights = score_weights_tensor.to(device=device, dtype=dtype)

    starting_epoch = 0
    if os.path.isfile(checkpoint_path):
        logging.info('Loading checkpoint from {0}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        starting_epoch = checkpoint['epoch'] + 1
    else:
        logging.info('No valid checkpoint is provided. Start to train from scratch...')
        model.apply(weights_init)

    # create checkpoint dir
    checkpoint_dir = 'checkpoints/{}'.format(experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_logger = Logger('logs/{}/train'.format(experiment_name))
    val_logger = Logger('logs/{}/val'.format(experiment_name))

    step = 0

    for e in range(starting_epoch, epochs):
        logging.info('Epoch {}'.format(e))

        for t, data in enumerate(loader_train):
            step += 1

            model.train()  # put model to training mode

            x = data['image'].to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = data['target'].to(device=device, dtype=torch.long)  # y is not a int value here; also an image

            optimizer.zero_grad()

            scores = model(x)
            loss = F.cross_entropy(scores, y, score_weights)
            # loss = F.cross_entropy(scores, y)

            loss.backward()
            optimizer.step()

            if (step + 1) % print_every == 0:
                logging.info('Epoch %d, step %d, training loss = %.4f' % (e, step, loss.item()))

                # logging for TensorBoard
                # 1. log scalar values (scalar summary)
                _, preds = scores.max(1)
                accuracy = (y == preds).float().mean()

                info = {'loss': loss.item(), 'accuracy': accuracy.item()}
                for tag, value in info.items():
                    train_logger.scalar_summary(tag, value, step + 1)

                # 2. log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    train_logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)
                    train_logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)

                # 3. log training images (image summary)
                info = {'train_images': x[:5].cpu().numpy()}
                for tag, images in info.items():
                    train_logger.image_summary(tag, images, step + 1)

        # save checkpoint every epoch
        checkpoint_path = os.path.join(checkpoint_dir, 'unet_checkpoint_epoch{}_{}.pth.tar'.format(e, strftime("%Y-%m-%d-%H-%M-%S", localtime())))
        logging.info('Saving to checkoutpoint file at {}'.format(checkpoint_path))
        save_checkpoint({
            'epoch': e,
            'arch': 'UNet',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, checkpoint_path)

        logging.info('Val accuracy at epoch {}'.format(e))
        val_loss, val_acc = check_accuracy(loader_val, model)  # loss is loss per sample


def check_accuracy(loader, model):
    """Evaluate the model on samples in the loader, and prints accuracy."""
    print('Calculating val set performance...')

    model.eval()  # put model to evaluation mode
    score_weights = score_weights_tensor.to(device=device, dtype=dtype)
    acc = 0
    loss_set = 0

    with torch.no_grad():
        num_correct = 0
        num_samples = 0

        for t, data in enumerate(loader):
            x = data['image'].to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = data['target'].to(device=device, dtype=torch.long)
            scores = model(x)

            loss = F.cross_entropy(scores, y, score_weights)
            loss_set += loss.item()
            # DEBUG logging.info('Val loss = %.4f' % loss.item())

            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples
        loss_per_sample = loss_set / num_samples

        logging.info('Got %d / %d correct, accuracy (%.2f)' % (num_correct, num_samples, 100 * acc))
        logging.info('Loss per sample is (%.2f)' % (loss_per_sample))

    return loss_per_sample, acc


def save_checkpoint(state, path='../checkpoints/checkpoints.pth.tar'):
    torch.save(state, path)


def main():
    num_classes = 3

    # larger model
    if model_choice == 'unet':
        model = Unet(feature_scale=feature_scale, n_classes=num_classes, is_deconv=True, in_channels=3, is_batchnorm=True)
    # year 2 best solution XD_XD's model, as the baseline model
    elif model_choice == 'unet_baseline':
        model = UnetBaseline(feature_scale=feature_scale, n_classes=num_classes, is_deconv=True, in_channels=3, is_batchnorm=True)
    else:
        sys.exit('Invalid model_choice {}, choose unet_baseline or unet'.format(model_choice))

    # can also use Nesterov momentum in optim.SGD
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate,
    #                     momentum=0.9, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, optimizer, epochs=total_epochs, print_every=print_every,
                checkpoint_path=starting_checkpoint_path)


if __name__ == '__main__':
    main()
