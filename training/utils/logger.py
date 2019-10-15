# Adapted from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
# Reference https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

import os
from io import BytesIO  # Python 3.x

import tensorflow as tf
import numpy as np
from PIL import Image


class Logger(object):

    def __init__(self, split, log_dir, aml_run):
        """Create a summary writer logging to log_dir."""
        log_dir = os.path.join(log_dir, split)
        self.writer = tf.summary.FileWriter(log_dir)

        self.split = split
        self.aml_run = aml_run

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

        self.aml_run.log('{}/{}'.format(self.split, tag), value)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img_np in enumerate(images):
            # Write the image to a buffer
            s = BytesIO()

            # torch image: C X H X W
            # numpy image: H x W x C
            img_np = img_np.transpose((1, 2, 0))
            im = Image.fromarray(img_np.astype(np.uint8))
            im.save(s, format='png')

            # Create an Image object
            img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img_np.shape[0],
                                       width=img_np.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_summary))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()