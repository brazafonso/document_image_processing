'''Script to rotate document image''' 

import argparse
import os

import cv2
from document_image_utils.image import rotate_image


def process_args():
    '''Use argparse to parse arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('image',                            type=str,                                   help='Image to rotate.')
    parser.add_argument('-lq', '--line_quantization',       type=int, default=None,                     help='Line quantization.')
    parser.add_argument('-d','--direction',                 type=str, default='auto',                   help='Direction of skew.')
    parser.add_argument('-ac','--auto_crop',                        action='store_false', default=True, help='Auto crop image (just for calculating rotation direction).')
    parser.add_argument('-cl','--crop_left',                type=int, default=None,                     help='Crop from left of image.')
    parser.add_argument('-cr','--crop_right',               type=int, default=None,                     help='Crop from right of image.')
    parser.add_argument('-ct','--crop_top',                 type=int, default=None,                     help='Crop from top of image.')
    parser.add_argument('-cb','--crop_bottom',              type=int, default=None,                     help='Crop from bottom of image.')
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

    rotated_image = rotate_image(image=image_path,
                                 line_quantetization=args.line_quantization,
                                 direction=args.direction,
                                 crop_left=args.crop_left,
                                 crop_right=args.crop_right,
                                 crop_top=args.crop_top,
                                 crop_bottom=args.crop_bottom,
                                 auto_crop=args.auto_crop,
                                 debug=args.debug)

    # save output
    output_path = args.output
    if output_path is None:
        dir = os.path.dirname(os.path.realpath(image_path))
        output_path = os.path.join(dir, 'rotated.png')

    if args.logs:
        print(f'Output: {output_path}')

    cv2.imwrite(output_path,rotated_image)
    
    