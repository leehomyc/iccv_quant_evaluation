"""Quantitative evaluation for ICCV 2017 paper."""
import logging
import os

import click
import numpy as np
import skimage.io
import skimage.measure

# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


@click.command()
@click.option('--syn_folder',
              type=click.STRING,
              default='syn',
              help='The synthesize image.')
@click.option('--gt_folder',
              type=click.STRING,
              default='gt',
              help='The ground truth.')
@click.option('--ext',
              type=click.STRING,
              default='.png',
              help='The image format.')
def compare(syn_folder, gt_folder, ext):
    """Gets a list of summary ops to run during training.

          Args:
              syn_folder: The folder with synthesized images.
              gt_folder: The folder with ground truth.
              ext: The extension

          Returns:
              Returns: None
      """
    avg_l1 = 0
    avg_l2 = 0
    avg_ssim = 0
    total_num = 0
    for subdir, dirs, files in os.walk(syn_folder):
        for img_file in files:
            syn_name = str(img_file)
            if ext in syn_name:
                total_num += 1

                gt_name = os.path.join(gt_folder, syn_name)
                syn_im = skimage.io.imread(syn_name)
                gt_im = skimage.io.imread(gt_name)

                # Compare L1
                l1_err = np.sum(np.fabs(syn_im.astype("float") -
                                        gt_im.astype("float")))
                # Normalize
                l1_err /= 255 * 3
                avg_l1 += l1_err

                # Compare L2
                l2_err = np.sum(
                    (syn_im.astype("float") - gt_im.astype("float")) ** 2,
                )
                # Normalize
                l2_err /= float(255 * 255) * 3
                avg_l2 += l2_err

                # Compare SSIM
                ssim_err = skimage.measure.compare_ssim(
                    syn_im, gt_im, multichannel=True)
                avg_ssim = avg_ssim + ssim_err

                logger.info('l1_err: {}. l2_err: {}. SSIM_err: {}'.format(
                    str(l1_err), str(l2_err), str(ssim_err)))
    avg_l1 = avg_l1 / total_num
    avg_l2 = avg_l2 / total_num
    avg_ssim = avg_ssim / total_num

    logger.info('avg_l1_err:{}. avg_l2_err:{}. avg_SSIM_err:{}'.format(
        str(avg_l1), str(avg_l2), str(avg_ssim)))


# Usage: python -m compare_images --syn_folder='folder_a'
# --gt_folder='folder_b' --ext='.png'
if __name__ == '__main__':
    compare()
