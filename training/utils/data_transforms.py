import torch

# referencing: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))  # ONLY transform image, not the target which is 256 x 256

        return {
                    'image': torch.from_numpy(image),
                    'target': torch.from_numpy(target),
                    'image_name': sample['image_name']
                }