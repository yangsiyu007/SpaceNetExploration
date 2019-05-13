import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_sample_img_gt(train_samples, val_samples, logger_train, logger_val):
    logger_train.image_summary('gt_train', train_samples, 1)
    logger_val.image_summary('gt_val', val_samples, 1)


colour_map = np.asarray([[0.7,0.7,1],[0.8, 1, 0.8],[0.9,0.6,0.1]], dtype=np.float32)
def render(softmax, hard=True):
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

