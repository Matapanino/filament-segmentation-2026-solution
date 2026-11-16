"""Multiscale IoU (MIoU) — VENDORED from the metric authors' reference code.

Provenance
----------
- Upstream: https://bitbucket.org/gsudmlab/multiscale_iou
  commit e423f4e4b0463c8267ba1d4d4f172be3e7f075ff (2021-08-08)
- License: MIT (see below). Written by Azim Ahmadzadeh (Georgia State
  University) — also an organizer of this competition; MIoU is the organizers'
  own metric for fine-structure segmentation (arXiv:2105.14572, ICIP 2021).
- We vendor rather than reimplement so our diagnostic numbers are the
  reference implementation's numbers, bit for bit (verified against the
  upstream code on synthetic masks in ``tests/test_msiou.py``).

Modifications (all mechanical; numerics untouched, parity-tested)
-----------------------------------------------------------------
1. Merged ``miou/edge_detection/sobel.py``, ``miou/utils/segmentation.py``,
   ``miou/metrics/metric.py`` and ``miou/metrics/miou.py`` into this single
   module; package-absolute imports rewritten accordingly.
2. Dropped imports the merged code never uses (``matplotlib``, ``PIL``).
3. ``sklearn.preprocessing.MinMaxScaler`` replaced by the identical
   ``(x - min) / (max - min)`` transform (drops the scikit-learn dependency).
4. ``numpy.trapz`` -> ``numpy.trapezoid`` (renamed in numpy 2.0; same
   function), with a fallback for numpy < 2.
5. Dropped the unused ``integral_simpson`` helper (its ``scipy.integrate.simps``
   import no longer exists in scipy >= 1.14).

Original license text
---------------------
- Multiscale IoU -

Licensed under the MIT License. (see LICENSE for details)
Written by Azim Ahmadzadeh (Georgia State University)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import abc
import copy

import numpy as np
from scipy import ndimage
from scipy.stats import linregress

try:  # numpy >= 2.0
    from numpy import trapezoid as _trapezoid
except ImportError:  # pragma: no cover - numpy < 2
    from numpy import trapz as _trapezoid


# --------------------------------------------------------------------------
# miou/edge_detection/sobel.py (verbatim)
# --------------------------------------------------------------------------
def get_edges(mask: np.ndarray):
    """
    This is a simple implementation of Sobel Edge Detection algorithm that
    is based on the Sobel filter from scipy.ndimage module.

    :param mask: a binary mask of an object whose edges are of interest.
    :return: a binary mask of 1's as edges and 0's as background.
    """
    sx = ndimage.sobel(mask, axis=0, mode='constant')
    sy = ndimage.sobel(mask, axis=1, mode='constant')
    sob: np.ndarray = np.hypot(sx, sy)
    sob[sob > 0] = 1
    sob[sob <= 0] = 0
    return sob


# --------------------------------------------------------------------------
# miou/utils/segmentation.py (verbatim, minus unused imports)
# --------------------------------------------------------------------------
class GridBasedSegmentation:
    """
    A simple class that does segmentation over a given image. In other words, it reduces the
    image-size by segmentizing the image into small cells and replacing each cell with 1 (if any
    non-zero value was present in that cell) or 0 (otherwise). To make the image size divisible
    by the given batch size, it does zero padding both horizontally and vertically.
    """

    def __init__(self, img, batch_size: int):
        self.img = copy.deepcopy(img)
        self.batch_size = int(batch_size)
        self._image_h = img.shape[0]
        self._image_w = img.shape[1]
        self._padding = [0, 0, 0, 0]  # top, bottom, left, right

    def _pad_with(self, vector, pad_width, iaxis, kwargs):
        """
        pads a 2-D array.

        This implementation is inspired from 'https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html'.

        - Usage::
            a = np.arange(6).reshape((2, 3))
            np.pad(a, ((1,2),(3,0)), pad_with, padder=0)  # ((top, bottom), (left, right))
        """
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        if pad_width[1] != 0:
            vector[-pad_width[1]:] = pad_value

    def shrink_by_segmentation(self, image):
        """
        divides the `image` into partitions using a gird-based segmentation (with each cell
        having sides of the size equal to `self.batch_size`), then replaces each cell with 1 if
        there is any non-zero value within that cell, and with 0, otherwise. The result will be a
        binary, 2D array, whose size is roughly equal to the size of the original image divide by
        `self.batch_size`. To make the image-width and image-height divisible by
        `self.batch_size` it adds some horizontal and/or vertical padding with zeros.

        :param image: the input 2-D array representing the image to be segmentized.
        :return: a 2D array of ones and zeros.
        """
        h = image.shape[0]
        w = image.shape[1]
        out = image.reshape(h // self.batch_size,
                            self.batch_size,
                            -1,
                            self.batch_size).swapaxes(1, 2).reshape(-1,
                                                                    self.batch_size,
                                                                    self.batch_size)
        s = np.einsum('ijk->i', out)
        s = s.reshape(h // self.batch_size, w // self.batch_size)
        s[s > 0] = 1
        return s

    def _pad_image(self):
        """
        Prepares the image in case any of its dimensions is not divisible by the class
        batch_size. It does so by added zero padding.
        """
        img_padded = self.img

        # if vertical padding is needed, add rows to 'top'
        if self._image_h % self.batch_size != 0:
            pad_height = self.batch_size - (self._image_h % self.batch_size)
            self._padding[0] = pad_height
            img_padded = np.pad(self.img,
                                ((pad_height, 0), (0, 0)),
                                self._pad_with, padder=0)
        # if horizontal padding is needed, add rows to 'left'
        if self._image_w % self.batch_size != 0:
            pad_width = self.batch_size - (self._image_w % self.batch_size)
            self._padding[2] = pad_width
            img_padded = np.pad(img_padded,
                                ((0, 0), (pad_width, 0)),
                                self._pad_with, padder=0)
        return img_padded

    def segmentize(self):
        """
        this methods wraps all the steps needs in box-counting: it pads the image if needed,
        """
        img_padded = self._pad_image()
        img_chunked = self.shrink_by_segmentation(img_padded)
        return img_chunked


# --------------------------------------------------------------------------
# miou/metrics/metric.py (verbatim)
# --------------------------------------------------------------------------
class IMetric(abc.ABC):
    @abc.abstractmethod
    def measure(self, mask1: np.ndarray, mask2: np.ndarray, *args, **kwargs):
        pass


# --------------------------------------------------------------------------
# miou/metrics/miou.py (verbatim, minus MinMaxScaler / trapz shims)
# --------------------------------------------------------------------------
class MIoU(IMetric):
    """
    This is the main class that implements MIoU.
    """
    def __init__(self, boxsizes, edge_only: bool = True):
        """
        Class constructor.

        :param boxsizes: a list or 1-D array of the side lengths to be used as the grid
        cell. The values of this collection determines the resolutions based on which
        MIoU is computed.

        :param edge_only: the default value is `True`. Usage of `False` is only experimental.
        """
        self.edge_only = edge_only
        self.boxsizes = np.array(boxsizes)
        self.aiou = 0
        self.box_counts_intersect = []
        self.box_counts_m1 = []
        self.ratios = []
        self.slope = 0
        self.area = 0

    def measure(self, mask1: np.ndarray, mask2: np.ndarray, *args, **kwargs):
        """
        computes the metric for the two given regions.

        :param mask1: mask segmentation of the reference (ground-truth) object.
        :param mask2: mask segmentation of the target (detected) object.
        """

        m1 = copy.deepcopy(mask1)
        m2 = copy.deepcopy(mask2)

        self.box_counts_intersect = []
        self.box_counts_m1 = []
        self.ratios = []

        if self.edge_only:
            m1 = get_edges(m1)
            m2 = get_edges(m2)

        for boxsize in self.boxsizes:
            gbs = GridBasedSegmentation(m1, boxsize)
            s1 = gbs.segmentize()
            gbs = GridBasedSegmentation(m2, boxsize)
            s2 = gbs.segmentize()

            # Box counting on m1 -------------------
            self.box_counts_m1.append(np.sum(s1) + 1)
            # --------------------------------------

            # Box counting on intersect(m1, m2) ----
            intersect = np.logical_and(s1, s2)
            self.box_counts_intersect.append(np.sum(intersect) + 1)
            # --------------------------------------

            self.ratios.append(self.box_counts_intersect[-1] / self.box_counts_m1[-1])

        normalized_boxsizes = normalize_boxsizes(self.boxsizes)
        self.slope, _, _, _, _ = linregress(normalized_boxsizes, self.ratios)
        self.aiou = np.abs(self.slope)

        self.area = integral_trapezoidal(self.ratios, dx=1/(len(self.boxsizes)-1))
        return self.area


def normalize_boxsizes(boxsizes) -> np.ndarray:
    """
    normalizes the values of `boxsizes` using a 0-1 transformation.

    :param boxsizes: a list or 1-D array of box sizes.
    :return: A 1-D array of transformed values.
    """
    b = np.asarray(boxsizes, dtype=float)
    return (b - b.min()) / (b.max() - b.min())


def integral_trapezoidal(y, dx):
    """
    Calculates area under the curves using Trapezoids.

    :param y: values of the curve on the Y axis.
    :param dx: spacing (for binning) on the X axis.

    :return: the area under the curve.
    """
    return _trapezoid(y, dx=dx)
