'''Script to auto crop document image''' 

import argparse
import os
import cv2
from document_image_utils.image import cut_document_margins
from document_image_utils.box import Box


def process_args():
    '''Use argparse to parse arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('image',                            type=str,                                   help='Image to crop.')
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

    cropped_area:Box = cut_document_margins(image=image_path,
                                        method=args.smoothing_method,
                                        logs=args.debug) 
    
    if args.logs:
        print(f'Cropped area: {cropped_area}')

    # crop image
    image = cv2.imread(image_path)
    cropped_image = image[cropped_area.top:cropped_area.bottom,cropped_area.left:cropped_area.right]

    # save output
    output_path = args.output
    if output_path is None:
        dir = os.path.dirname(os.path.realpath(image_path))
        output_path = os.path.join(dir, 'cropped.png')

    if args.logs:
        print(f'Output: {output_path}')

    cv2.imwrite(output_path,cropped_image)
    
    