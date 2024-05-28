'''Script to split document image into its columns''' 

import argparse
import os
import cv2
from document_image_utils.image import split_page_columns


def process_args():
    '''Use argparse to parse arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('image',                            type=str,                                   help='Image to rotate.')
    parser.add_argument('-sm', '--smoothing_method',        type=str, default='WhittakerSmoother',      help='Smoothing method.', choices=['WhittakerSmoother','savgol_filter'])
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

    column_images = split_page_columns(image_path=image_path,
                                        method=args.smoothing_method,
                                        logs=args.debug) 

    # save output
    output_path = args.output
    if output_path is None:
        dir = os.path.dirname(os.path.realpath(image_path))
        output_path = dir

    if args.logs:
        print(f'Output path: {output_path}')
        print(f'Number of columns: {len(column_images)}')

    for i, image in enumerate(column_images):
        cv2.imwrite(os.path.join(output_path, f'column_{i}.png'), image)
    
    