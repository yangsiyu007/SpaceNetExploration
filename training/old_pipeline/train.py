import argparse
import logging
import os
import shutil
import sys
from time import localtime, strftime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from tqdm import tqdm

# add the script directory to PYTHONPATH
# mount_root = os.environ.get('AZ_BATCHAI_JOB_MOUNT_ROOT', '')
# sys.path.insert(0, '{}/{}'.format(mount_root, 'afs_spacenet/scripts'))
# configurations for the training run
import train_dist_config as train_config

from models.unet.unet import Unet
from models.unet.unet_baseline import UnetBaseline
from utils.data_transforms import ToTensor
from utils.dataset import SpaceNetDataset
from utils.logger import Logger
from utils.train_utils import AverageMeter, log_sample_img_gt, render


"""
train.py

Execute from the root directory SpaceNetExploration to save checkpoints and logs there.

It requires train_config.py to be in the path.

Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""


logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logging.info('Using PyTorch version %s.', torch.__version__)

# distributed training settings parsed through argparse
parser = argparse.ArgumentParser(description='UNet training on the SpaceNet dataset')
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist_backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='rank of the worker')

# config for the run

# make data_path_val = data_path_test if you want to evaluate on the test set
evaluate_only = train_config.TRAIN['evaluate_only']

use_gpu = train_config.TRAIN['use_gpu']
dtype = train_config.TRAIN['dtype']

torch.backends.cudnn.benchmark = True  # enables benchmark mode in cudnn, see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936

data_path_root = train_config.TRAIN['data_path_root']
data_path_train = os.path.join(data_path_root, train_config.TRAIN['data_path_train'])
data_path_val = os.path.join(data_path_root, train_config.TRAIN['data_path_val'])
data_path_test = os.path.join(data_path_root, train_config.TRAIN['data_path_test'])

tensorboard_path = train_config.TRAIN['tensorboard_path']
out_checkpoint_dir = train_config.TRAIN['out_checkpoint_dir']

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
loss_weights = torch.from_numpy(np.array(train_config.TRAIN['loss_weights']))
learning_rate = train_config.TRAIN['learning_rate']
print_every = train_config.TRAIN['print_every']
total_epochs = train_config.TRAIN['total_epochs']

split_tags = ['trainval', 'test']  # compatibility with the SpaceNet image preparation code - do not change


# device configuration
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
logging.info('Using device: %s.', device)


def get_sample_images(loader, which_set='train'):
    # which_set could be 'train' or 'val'; loader should already have shuffled them; gets one batch
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
                'Device rank {}, Epoch {}, step {}, train minibatch_loss is {}, train minibatch_accuracy is {}'.format(
                    args.rank, epoch, step, info['minibatch_loss'], info['minibatch_accuracy']))

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
    global args, sample_images_train_tensors, sample_images_val_tensors
    args = parser.parse_args()
    print('args.world_size: ', args.world_size)
    print('args.dist_backend: ', args.dist_backend)
    print('args.rank: ', args.rank)

    # more info on distributed PyTorch see https://pytorch.org/tutorials/intermediate/dist_tuto.html
    args.distributed = args.world_size >= 2
    print('is distributed: '.format(args.distributed))
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print('dist.init_process_group() finished.')

    # data sets and loaders
    dset_train = SpaceNetDataset(data_path_train, split_tags, transform=T.Compose([ToTensor()]))
    dset_val = SpaceNetDataset(data_path_val, split_tags, transform=T.Compose([ToTensor()]))
    logging.info('Training set size: {}, validation set size: {}'.format(
        len(dset_train), len(dset_val)))

    # need to instantiate these data loaders to produce the sample images because they need to be shuffled!
    loader_train = DataLoader(dset_train, batch_size=train_batch_size, shuffle=True,
                              num_workers=num_workers)  # shuffle True to reshuffle at every epoch

    loader_val = DataLoader(dset_val, batch_size=val_batch_size, shuffle=True,
                            num_workers=num_workers)

    # get one batch of sample images that are used to visualize the training progress throughout this run
    sample_images_train, sample_images_train_tensors = get_sample_images(loader_train, which_set='train')
    sample_images_val, sample_images_val_tensors = get_sample_images(loader_val, which_set='val')

    if args.distributed:
        # re-instantiate the training data loader to make distributed training possible
        train_batch_size_dist = train_batch_size * args.world_size
        logging.info('Using train_batch_size_dist {}.'.format(train_batch_size_dist))
        train_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.distributed.DistributedSampler(dset_train),
            batch_size=train_batch_size_dist, drop_last=False)
        # TODO https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        # check if need num_replicas and rank
        print('train_sampler created successfully.')
        loader_train = DataLoader(dset_train, num_workers=num_workers,
                                  pin_memory=True, batch_sample=train_sampler)

        loader_val = DataLoader(dset_val, batch_size=val_batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        print('both data loaders created successfully.')

    # checkpoint dir
    checkpoint_dir = out_checkpoint_dir

    logger_train = Logger('{}/train'.format(tensorboard_path))
    logger_val = Logger('{}/val'.format(tensorboard_path))
    log_sample_img_gt(sample_images_train, sample_images_val, logger_train, logger_val)
    logging.info('Logged ground truth image samples')

    num_classes = 3

    # larger model
    if model_choice == 'unet':
        model = Unet(feature_scale=feature_scale, n_classes=num_classes, is_deconv=True, in_channels=3, is_batchnorm=True)
    # year 2 best solution XD_XD's model, as the baseline model
    elif model_choice == 'unet_baseline':
        model = UnetBaseline(feature_scale=feature_scale, n_classes=num_classes, is_deconv=True, in_channels=3, is_batchnorm=True)
    else:
        sys.exit('Invalid model_choice {}, choose unet_baseline or unet'.format(model_choice))
    print('model instantiated.')

    if not args.distributed:
        model = model.to(device=device, dtype=dtype)  # move the model parameters to target device
        #model = torch.nn.DataParallel(model).cuda() # Batch AI example
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        print('torch.nn.parallel.DistributedDataParallel() ran.')

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

    # run training or evaluation
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
        # log the val images too

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
