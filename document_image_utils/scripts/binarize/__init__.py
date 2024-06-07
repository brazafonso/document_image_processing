'''Script to binarize image''' 

import argparse
import os
import cv2
import numpy as np
from document_image_utils.image import binarize
from document_image_utils.box import Box


def process_args():
    '''Use argparse to parse arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('image',                            type=str,                                   help='Image to rotate.')
    parser.add_argument('-ds', '--denoise_strength',        type=int, default=None,                     help='Denoise strength. If auto, calculates SNR of image and chooses the best denoise strength.')
    parser.add_argument('-1bpp', '--one_bit_per_pixel',     action='store_true',                        help='Convert to one bit per pixel.')
    parser.add_argument('-i','--invert',                    action='store_true',                        help='Invert image binarization (black background and white text).')
    parser.add_argument('-o','--output',                    type=str, default=None,                     help='Output path.')
    parser.add_argument('-l', '--logs',                             action='store_false',               help='Print logs.')
    parser.add_argument('--debug',                                  action='store_true',                help='Debug mode.')
    args = parser.parse_args()
    return args





def main():
    args = process_args()

    image_path = args.image
    # fix image path if not absolute
    if os.path.dirname(image_path) == '':
        image_path = os.path.join(os.getcwd(), image_path)

    if args.logs:
        print(f'Image: {image_path}')

    binarized_image = binarize(image=image_path,
                                denoise_strength=args.denoise_strength,
                                invert=args.invert,
                                logs=args.debug)


    # save output
    output_path = args.output
    if output_path is None:
        dir = os.path.dirname(os.path.realpath(image_path))
        output_path = os.path.join(dir, 'binarized.png')

    if args.logs:
        print(f'Output: {output_path}')

    params = []
    if args.one_bit_per_pixel:
        params = [cv2.IMWRITE_PNG_BILEVEL, 1]

    cv2.imwrite(output_path,binarized_image,params=params)
    
    