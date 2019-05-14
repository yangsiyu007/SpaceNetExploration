import argparse
import logging
import os
import shutil
import sys
from time import localtime, strftime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import train_single_gpu_config as train_config
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from tqdm import tqdm

from models.unet.unet import Unet
from models.unet.unet_baseline import UnetBaseline
from utils.data_transforms import ToTensor
from utils.dataset import SpaceNetDataset
from utils.logger import Logger
from utils.train_utils import AverageMeter, log_sample_img_gt, render

"""
train.py

It requires train_config.py to be in the path.

Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""


logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logging.info('Using PyTorch version %s.', torch.__version__)


parser = argparse.ArgumentParser(description='UNet training on the SpaceNet dataset')
parser.add_argument('--experiment_name', type=str, default=None,
                    help='name of the experiment; outputs will be placed in a folder named this')
parser.add_argument('--data_path_root', type=str, default=None,
                    help='path to the root of data folders')
parser.add_argument('--out_dir', type=str,
                    help=('Path to directory to save checkpoints and logs in; a folder named experiment_name '
                          'will be created there.'))
args = parser.parse_args()

# config for the run
evaluate_only = train_config.TRAIN['evaluate_only']

use_gpu = train_config.TRAIN['use_gpu']
dtype = train_config.TRAIN['dtype']

torch.backends.cudnn.benchmark = True  # enables benchmark mode in cudnn, see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936

data_path_root = train_config.TRAIN['data_path_root'] if args.data_path_root is None else args.data_path_root
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
starting_checkpoint_path = train_config.TRAIN['starting_checkpoint_path']

# weights for computing the loss function; absolute values of the weights do not matter
# [background, interior of building, border of building]
loss_weights = torch.from_numpy(np.array(train_config.TRAIN['loss_weights']))
learning_rate = train_config.TRAIN['learning_rate']
print_every = train_config.TRAIN['print_every']
total_epochs = train_config.TRAIN['total_epochs']

experiment_name = train_config.TRAIN['experiment_name'] if args.experiment_name is None else args.experiment_name

split_tags = ['trainval', 'test']  # compatibility with the SpaceNet image preparation code - do not change


# device configuration
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
logging.info('Using device: %s.', device)

# data sets and loaders
dset_train = SpaceNetDataset(data_path_train, split_tags, transform=T.Compose([ToTensor()]))
loader_train = DataLoader(dset_train, batch_size=train_batch_size, shuffle=True,
                          num_workers=num_workers)  # shuffle True to reshuffle at every epoch

dset_val = SpaceNetDataset(data_path_val, split_tags, transform=T.Compose([ToTensor()]))
loader_val = DataLoader(dset_val, batch_size=val_batch_size, shuffle=True,
                        num_workers=num_workers)  # also reshuffle val set because loss is recorded for the last batch

dset_test = SpaceNetDataset(data_path_test, split_tags, transform=T.Compose([ToTensor()]))
loader_test = DataLoader(dset_test, batch_size=test_batch_size, shuffle=True,
                         num_workers=num_workers)

logging.info('Training set size: {}, validation set size: {}, test set size: {}'.format(
    len(dset_train), len(dset_val), len(dset_test)))


def get_sample_images(which_set='train'):
    # which_set could be 'train' or 'val'; loader should already have shuffled them; gets one batch
    loader = loader_train if which_set == 'train' else loader_val
    images = None
    image_tensors = None
    for batch in loader:
        image_tensors = batch['image']
        images = batch['image'].cpu().numpy()
        break  # take the first shuffled batch
    images_li = []
    for b in range(0, images.shape[0]):
        images_li.append(images[b, :, :, :])
    return images_li, image_tensors

sample_images_train, sample_images_train_tensors = get_sample_images(which_set='train')
sample_images_val, sample_images_val_tensors = get_sample_images(which_set='val')


def visualize_result_on_samples(model, sample_images, logger, step, split='train'):
    model.eval()
    with torch.no_grad():
        sample_images = sample_images.to(device=device, dtype=dtype)
        scores = model(sample_images).cpu().numpy()
        images_li = []
        for i in range(scores.shape[0]):
            input = scores[i, :, :, :].squeeze()
            picture = render(input)
            images_li.append(picture)

        logger.image_summary('result_{}'.format(split), images_li, step)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()


def train(loader_train, model, criterion, optimizer, epoch, step, logger_train):
    for t, data in enumerate(tqdm(loader_train)):
        # put model to training mode; we put it in eval mode in visualize_result_on_samples for every print_every
        model.train()
        step += 1
        x = data['image'].to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = data['target'].to(device=device, dtype=torch.long)  # y is not a int value here; also an image

        # forward pass on this batch
        scores = model(x)
        loss = criterion(scores, y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TensorBoard logging and print a line to stdout; note that the accuracy is wrt the current mini-batch only
        if step % print_every == 1:
            # 1. log scalar values (scalar summary)
            _, preds = scores.max(1)
            accuracy = (y == preds).float().mean()

            info = {'minibatch_loss': loss.item(), 'minibatch_accuracy': accuracy.item()}
            for tag, value in info.items():
                logger_train.scalar_summary(tag, value, step + 1)

            logging.info(
                'Epoch {}, step {}, train minibatch_loss is {}, train minibatch_accuracy is {}'.format(
                    epoch, step, info['minibatch_loss'], info['minibatch_accuracy']))

            # 2. log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger_train.histo_summary(tag, value.data.cpu().numpy(), step + 1)
                logger_train.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)

            # 3. log training images (image summary)
            visualize_result_on_samples(model, sample_images_train_tensors, logger_train, step, split='train')
            visualize_result_on_samples(model, sample_images_val_tensors, logger_train, step, split='val')

    return step


def evaluate(loader, model, criterion):
    """Evaluate the model on dataset of the loader"""
    losses = AverageMeter()
    accuracies = AverageMeter()

    model.eval()  # put model to evaluation mode
    with torch.no_grad():
        for t, data in enumerate(loader):
            x = data['image'].to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = data['target'].to(device=device, dtype=torch.long)
            scores = model(x)

            loss = criterion(scores, y)
            # DEBUG logging.info('Val loss = %.4f' % loss.item())

            _, preds = scores.max(1)
            accuracy = (y == preds).float().mean()

            losses.update(loss.item(), x.size(0))
            accuracies.update(accuracy.item(), 1)  # average already taken for accuracy for each pixel

    return losses.avg, accuracies.avg


def save_checkpoint(state, is_best, path='../checkpoints/checkpoints.pth.tar', checkpoint_dir='../checkpoints'):
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def main():
    num_classes = 3

    # create checkpoint dir
    out_dir = '../out_dir' if args.out_dir is None else args.out_dir
    checkpoint_dir = os.path.join(out_dir, experiment_name, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger_train = Logger(os.path.join(out_dir, experiment_name, 'logs', 'train'))
    logger_val = Logger(os.path.join(out_dir, experiment_name, 'logs', 'val'))
    log_sample_img_gt(sample_images_train, sample_images_val, logger_train, logger_val)
    logging.info('Logged ground truth image samples')

    # larger model
    if model_choice == 'unet':
        model = Unet(feature_scale=feature_scale, n_classes=num_classes, is_deconv=True,
                     in_channels=3, is_batchnorm=True)
    # year 2 best solution XD_XD's model, as the baseline model
    elif model_choice == 'unet_baseline':
        model = UnetBaseline(feature_scale=feature_scale, n_classes=num_classes, is_deconv=True,
                             in_channels=3, is_batchnorm=True)
    else:
        sys.exit('Invalid model_choice {}, choose unet_baseline or unet'.format(model_choice))

    model = model.to(device=device, dtype=dtype)  # move the model parameters to CPU/GPU

    criterion = nn.CrossEntropyLoss(weight=loss_weights).to(device=device, dtype=dtype)

    # can also use Nesterov momentum in optim.SGD
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate,
    #                     momentum=0.9, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # resume from a checkpoint if provided
    starting_epoch = 0
    best_acc = 0.0

    if os.path.isfile(starting_checkpoint_path):
        logging.info('Loading checkpoint from {0}'.format(starting_checkpoint_path))
        checkpoint = torch.load(starting_checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        starting_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0.0)
    else:
        logging.info('No valid checkpoint is provided. Start to train from scratch...')
        model.apply(weights_init)

    if evaluate_only:
        val_loss, val_acc = evaluate(loader_val, model, criterion)
        print('Evaluated on val set, loss is {}, accuracy is {}'.format(val_loss, val_acc))
        return

    step = starting_epoch * len(dset_train)

    for epoch in range(starting_epoch, total_epochs):
        logging.info('Epoch {} of {}'.format(epoch, total_epochs))

        # train for one epoch
        step = train(loader_train, model, criterion, optimizer, epoch, step, logger_train)

        # evaluate on val set
        logging.info('Evaluating model on the val set at the end of epoch {}...'.format(epoch))
        val_loss, val_acc = evaluate(loader_val, model, criterion)
        logging.info('\nEpoch {}, val loss is {}, val accuracy is {}\n'.format(epoch, step, val_loss, val_acc))
        logger_val.scalar_summary('val_loss', val_loss, step + 1)
        logger_val.scalar_summary('val_acc', val_acc, step + 1)
        # TODO log the val images too

        # record the best accuracy; save checkpoint for every epoch
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        checkpoint_path = os.path.join(checkpoint_dir,
                                       'checkpoint_epoch{}_{}.pth.tar'.format(epoch, strftime("%Y-%m-%d-%H-%M-%S", localtime())))
        logging.info(
            'Saving to checkoutpoint file at {}. Is it the highest accuracy checkpoint so far: {}'.format(
                checkpoint_path, str(is_best)))
        save_checkpoint({
            'epoch': epoch + 1,  # saved checkpoints are numbered starting from 1
            'arch': model_choice,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc
        }, is_best, checkpoint_path, checkpoint_dir)


if __name__ == '__main__':
    main()
