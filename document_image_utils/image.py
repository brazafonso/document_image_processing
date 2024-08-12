import os
import re
from typing import Union
import cv2
from .box import *
from scipy import ndimage
import numpy as np
from scipy.signal import *
from whittaker_eilers import WhittakerSmoother
from matplotlib import pyplot as plt

file_path = os.path.dirname(os.path.realpath(__file__))

def get_concat_h(im1, im2,margin=0):
    '''Concatenate images horizontally'''
    dst = np.zeros((im1.shape[0], im1.shape[1] + im2.shape[1] + margin, 3), dtype=np.uint8)
    dst[:, :im1.shape[1], :] = im1
    dst[:, im1.shape[1] + margin:, :] = im2
    return dst


def split_page_columns(image_path:str,columns:Union[list[Box],None]=None,
                       method:str='WhittakerSmoother',logs:bool=False)->list[cv2.typing.MatLike]:
    '''Split image into columns images'''
    image = cv2.imread(image_path)
    columns_image = []

    # if no columns are given, divide columns
    if not columns:
        columns = divide_columns(image_path,method=method,logs=logs)

    for column in columns:
        columns_image.append(image[column.top:column.bottom,column.left:column.right])
    return columns_image


def concatentate_columns(columns):
    '''Concatenate columns images horizontally in a single image'''
    image = None
    if columns:
        image = columns[0]
        for column in columns[1:]:
            image = get_concat_h(image,column,15)
    return image



def black_and_white(image_path):
    '''Convert image to black and white'''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return thresh


def get_image_info(image_path:str)->Box:
    '''Get image info'''
    image = cv2.imread(image_path)
    image_info = Box(0,len(image[0]),0,len(image))
    return image_info



def calculate_dpi(image_info:Box,dimensions:Box)->float:
    '''Calculate dpi'''
    dpi = (image_info.width/dimensions.width + image_info.height/dimensions.height) / 2
    return dpi



def create_vertical_aligned_pixel_set(pixels:list,image_shape:tuple,direction:str='clockwise'):
    '''Create pixel set
    
    Tries to create a set of pixels that are vertically aligned, with no great x variance
    Also does not add pixels that are too far apart from each other (using image shape)'''
    pixel_set = [pixels[0]]
    pixel_set_x_var_sum = 0
    for i in range(1,len(pixels)):
        if (direction == 'clockwise' and pixels[i][0] < pixel_set[-1][0]) or (direction == 'counter_clockwise' and pixels[i][0] > pixel_set[-1][0]):
            # check x distance relative to image shape
            if abs(pixels[i][0] - pixel_set[-1][0])/image_shape[1] <= 0.1:
                cw_set_x_avg = pixel_set_x_var_sum/(len(pixel_set))
                # check x variance
                if not pixel_set_x_var_sum or (abs(pixels[i][0] - pixel_set[-1][0]) <= cw_set_x_avg):
                    pixel_set_x_var_sum += abs(pixels[i][0] - pixel_set[-1][0])
                    pixel_set.append(pixels[i])
        # for same x coordinate, height difference cant be more than 5% of image height
        elif direction == 'none' and pixels[i][0] == pixel_set[-1][0] and (abs(pixels[i][1] - pixel_set[-1][1])/image_shape[0] <= 0.05):
            pixel_set.append(pixels[i])
    return pixel_set


def calculate_rotation_direction(image:Union[str,cv2.typing.MatLike],line_quantetization:int=200,crop_left:int=None,crop_right:int=None,crop_top:int=None,crop_bottom:int=None,debug:bool=False):
    '''Calculate rotation direction (counter-clockwise or clockwise)
    
    On left margin of image compare the groups of ordered black pixels by x coordinate
    If the largest group is x descending (from top to bottom) the direction is clockwise, else counter-clockwise
    If largest group is of same x coordinate, the direction is none'''

    if isinstance(image,str):
        image = cv2.imread(image)

    if crop_left is None:
        crop_left = round(image.shape[1]*0.01)
    if crop_right is None:
        crop_right = round(image.shape[1]*0.01)
    if crop_top is None:
        crop_top = round(image.shape[0]*0.01)
    if crop_bottom is None:
        crop_bottom = round(image.shape[0]*0.01)


    if debug:
        test_path = '/'.join(image.split('/')[:-1]) if isinstance(image,str) else '.'
        if not os.path.exists(f'{test_path}/test'):
            os.mkdir(f'{test_path}/test')
        test_img_name = image.split("/")[-1] if isinstance(image,str) else 'test'
        test_path = f'{test_path}/test/{test_img_name}'


    direction = 'None'
    # crop margin
    image = image[crop_top:image.shape[0]-crop_bottom,crop_left:image.shape[1]-crop_right]
    binarized = binarize_fax(image,invert=True,logs=False)
    filtered = cv2.medianBlur(binarized, 3)
    dilation = cv2.dilate(filtered, np.ones((0,20),np.uint8),iterations=3)
    transformed_image = dilation

    if debug:
        cv2.imwrite(f'{test_path}_thresh.png',binarized)
        cv2.imwrite(f'{test_path}_filtered.png',filtered)
        cv2.imwrite(f'{test_path}_dilation.png',dilation)


    # calculate sets
    pixels = []
    step = math.floor(transformed_image.shape[0]/line_quantetization)

    # get pixels that are not black
    for y in range(0,transformed_image.shape[0], step):
        for x in range(transformed_image.shape[1]):
            if transformed_image[y][x] != 0:
                pixels.append((x,y))
                break

    if debug:
    # draw pixels
        copy_image = cv2.imread(f'{test_path}_dilation.png')
        for pixel in pixels:
            cv2.circle(copy_image, pixel, 7, (0,0,255), -1)
        cv2.imwrite(f'{test_path}_pixels.png',copy_image)

    # make list of sets
    # each set is a list of pixels in x coordinates order (ascending or descending depending on rotation direction)
    clockwise_sets = []
    counter_clockwise_sets = []
    same_x_sets = []
    for i in range(1,len(pixels)):
        new_cw_set = create_vertical_aligned_pixel_set(pixels[i:],transformed_image.shape,'clockwise')
        new_ccw_set = create_vertical_aligned_pixel_set(pixels[i:],transformed_image.shape,'counter_clockwise')
        new_same_x_set = create_vertical_aligned_pixel_set(pixels[i:],transformed_image.shape,'none')

        clockwise_sets.append(pixels_set_remove_outliers(new_cw_set,'clockwise'))
        counter_clockwise_sets.append(pixels_set_remove_outliers(new_ccw_set,'counter_clockwise'))
        same_x_sets.append(new_same_x_set)

   

    # find biggest sets
    biggest_clockwise_set = max(clockwise_sets, key=len) if clockwise_sets else []
    biggest_counter_clockwise_set = max(counter_clockwise_sets, key=len) if counter_clockwise_sets else []
    biggest_same_x_set = max(same_x_sets, key=len) if same_x_sets else []

    if debug:
        print('test','clockwise',len(biggest_clockwise_set))
        print('counter_clockwise',len(biggest_counter_clockwise_set))
        print('same_x',len(biggest_same_x_set))

    if debug:
        # draw biggest sets
        for pixel in biggest_clockwise_set:
            cv2.circle(image, pixel, 7, (0,0,255), -1)
        for pixel in biggest_counter_clockwise_set:
            cv2.circle(image, pixel, 7, (0,255,0), -1)
        for pixel in biggest_same_x_set:
            cv2.circle(image, pixel, 7, (255,0,0), -1)
        cv2.imwrite(f'{test_path}_biggest_sets.png',image)

    # check biggest set between clockwise, counter and same
    if len(biggest_clockwise_set) > len(biggest_counter_clockwise_set) and len(biggest_clockwise_set) > len(biggest_same_x_set):
        direction = 'clockwise'
    elif len(biggest_counter_clockwise_set) > len(biggest_clockwise_set) and len(biggest_counter_clockwise_set) > len(biggest_same_x_set):
        direction = 'counter_clockwise'
    else:
        direction = 'none'
    

    return direction


def pixels_set_remove_outliers(set:list,direction:str='clockwise'):
    '''Removes outliers from set'''

    aux_set = set
    removed_pixel = True
    # while outliers detected
    # remove outliers
    j = 0
    x_avg = 0
    while removed_pixel and len(aux_set) > 1:
        j+=1
        new_set = []

        # average displacement of x coordinates
        x_avg = 0
        for i in range(1,len(aux_set)):
            x1 = aux_set[i-1][0]
            x2 = aux_set[i][0]
            if direction == 'counter_clockwise':
                x1,x2 = x2,x1
            x_avg +=  x1 - x2
                
        x_avg = x_avg / (len(aux_set)-1)

        # remove outlier pixels, using average displacement
        for i in range(1,len(aux_set)):
            x1 = aux_set[i-1][0]
            x2 = aux_set[i][0]
            if direction == 'counter_clockwise':
                x1,x2 = x2,x1

            if abs(x1 - x2 - x_avg) <= x_avg:
                new_set.append(aux_set[i])

        x1 = aux_set[0][0]
        x2 = aux_set[1][0]
        if direction == 'counter_clockwise':
            x1,x2 = x2,x1
        #check first point
        if abs(x1 - x2 - x_avg) <= x_avg:
            new_set = [aux_set[0]] + new_set

        if len(new_set) == len(aux_set):
            removed_pixel = False
        aux_set = new_set

    # print('iterations',j,aux_set,x_avg)

    return aux_set


def rotate_image_alt(image):
    '''Rotate image alt, based on longest hough line'''
    test_path = image.split('/')[:-1]
    test_path = '/'.join(test_path)
    if not os.path.exists(f'{test_path}/test'):
        os.mkdir(f'{test_path}/test')
    test_path = f'{test_path}/test/{image.split("/")[-1]}'

    img_before = cv2.imread(image)

    # crop image (remove all margins to leave center)
    img_before = img_before[100:img_before.shape[0]-100, 200:img_before.shape[1]-200] 
    
    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    cv2.imwrite(test_path+'_edges.png', img_edges)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=10)
    
    # draw lines on image
    all_lines_img = cv2.imread(image)
    if (lines is not None):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(all_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.imwrite(test_path+'_all_lines.png', all_lines_img)


    image_info = get_image_info(image)
    # get longest line
    longest_line = None
    longest_line_distance = 0
    if (lines is not None):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # not border
            if (x1 == image_info.left or x1 == image_info.right or x2 == image_info.left or x2 == image_info.right):
                continue
            line_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if (longest_line is None):
                longest_line = (x1, y1, x2, y2)
                longest_line_distance = line_distance
            elif (line_distance > longest_line_distance):
                longest_line = (x1, y1, x2, y2)
                longest_line_distance = line_distance
    
    if not longest_line:
        return
    
    # get angle
    angle = abs(math.degrees(math.atan2(longest_line[3] - longest_line[1], longest_line[2] - longest_line[0])))

    # if (median_angle >= 0):
    # 	img_rotated = ndimage.rotate(img_before, median_angle)
    # else:
    # 	img_rotated = ndimage.rotate(img_before, 180+median_angle)
    
    print("Angle is {}".format(angle))

    img = cv2.imread(image)
    # showImage(img_rotated)
    img_rotated = ndimage.rotate(img, 90-angle)
    
    cv2.imwrite(test_path+'_rotated_alt.png', img_rotated)

    # draw longest line
    cv2.line(img_before, (longest_line[0], longest_line[1]), (longest_line[2], longest_line[3]), (255, 0, 0), 3)
    cv2.imwrite(test_path+'_lines_alt.png', img_before)







def rotate_image(image:Union[str,cv2.typing.MatLike],line_quantetization:int=None,direction:str='auto',
                 crop_left:int=None,crop_right:int=None,crop_top:int=None,crop_bottom:int=None,auto_crop:bool=False,
                 debug:bool=False)->cv2.typing.MatLike:
    '''Finds the angle of the image and rotates it
    
    Based on the study by: W. Bieniecki, Sz. Grabowski, W. Rozenberg 
    
    Steps:
    1. Crop image
    2. Grey Scale image
    3. Binarize image
    4. For each line (y coordinate; taking steps according to line_quantetization)
        4.1 Get first black pixel in each line
    5. Calculate best list of sets of pixels
        5.1 Pixeis are ordered from left to right or right to left
    6. Remove outliers from set
    7. Find angle
    8. Rotate image
    '''

    test_path = os.getcwd()
    if not os.path.exists(f'{test_path}/test'):
        os.mkdir(f'{test_path}/test')
    test_path = f'{test_path}/test/test.png'
    
    if isinstance(image,str):
        og_img = cv2.imread(image)
    else:
        og_img = image

    # check if image is laied
    ## width > height
    ## rotate 90 degrees
    if og_img.shape[0] < og_img.shape[1]:
        og_img = cv2.rotate(og_img, cv2.ROTATE_90_CLOCKWISE)


    if not line_quantetization:
        line_quantetization = round(og_img.shape[0]*0.1)

    if not auto_crop:
        if crop_left is None:
            crop_left = round(og_img.shape[1]*0.01)
        if crop_right is None:
            crop_right = round(og_img.shape[1]*0.01)
        if crop_top is None:
            crop_top = round(og_img.shape[0]*0.01)
        if crop_bottom is None:
            crop_bottom = round(og_img.shape[0]*0.01)

        # crop margin
        cut_img = og_img[crop_top:og_img.shape[0] - crop_bottom, crop_left:og_img.shape[1] - crop_right]
    else:
        cropped = cut_document_margins(og_img.copy())
        cut_img = og_img[cropped.top:cropped.bottom, cropped.left:cropped.right]

    binary_img = binarize_fax(cut_img,invert=True)

    # get first black pixel in each line of image
    ## analyses lines acording to line_quantetization
    pixels = []
    step = math.floor(binary_img.shape[0]/line_quantetization)
    for y in range(0,binary_img.shape[0], step):
        for x in range(binary_img.shape[1]):
            if binary_img[y][x] != 0:
                pixels.append((x,y))
                break

    # estimate rotation direction
    if direction == 'auto' or direction not in ['clockwise', 'counter_clockwise']:
        direction = calculate_rotation_direction(cut_img.copy(), debug=debug)

    if debug:
        print('direction',direction)

    if direction == 'none':
        return og_img


    # make list of sets
    # each set is a list of pixels in x coordinates order (ascending or descending depending on rotation direction)
    sets = []
    for i in range(1,len(pixels)-1):
        new_set = create_vertical_aligned_pixel_set(pixels[i:], binary_img.shape, direction)
        sets.append(new_set)


    set = []
    # choose set with most elements
    for s in sets:
        if not set:
            set = s
        elif len(s) > len(set):
            set = s
    if debug:
        print('set',len(set))


    new_set = pixels_set_remove_outliers(set,direction)

    if len(new_set) < 2:
        return og_img
    
    # get extreme points
    left_most_point = new_set[0]
    right_most_point = new_set[-1]
    
    # find angle
    angle = math.degrees(math.atan((right_most_point[1] - left_most_point[1]) / (right_most_point[0] - left_most_point[0])))

    if debug:
        print('angle',angle)

    rotation_angle = 90 - abs(angle)
    if direction == 'counter_clockwise':
        rotation_angle = -rotation_angle

    h,w = og_img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
    rotated_img = cv2.warpAffine(og_img, rotation_matrix, (w, h),borderValue=(255, 255, 255))

    # do small adjustments using leptonica
    #     this method can't be used for adjusting more than a few degrees
    leptonica_adjust_rotation_path = f'{file_path}/leptonica_lib/adjust_rotation'
    if os.path.exists(leptonica_adjust_rotation_path):
        cv2.imwrite(f'leptonica_tmp.png',rotated_img)
        os.system(f'{leptonica_adjust_rotation_path} leptonica_tmp.png leptonica_tmp.png')
        rotated_img = cv2.imread(f'leptonica_tmp.png')
        os.remove(f'leptonica_tmp.png')


    ## test images
    if debug:
        cv2.imwrite(test_path + '_rotated.png', rotated_img)
        img = og_img
        # draw points from set
        for p in set:
            cv2.circle(img, (p[0]+50, p[1]), 7, (255, 0, 0), -1)

        cv2.imwrite(test_path + '_points_1.png', img)

        img = og_img

        # draw points from set
        for p in new_set:
            cv2.circle(img, (p[0]+50, p[1]), 7, (255, 0, 0), -1)

        cv2.imwrite(test_path + '_points.png', img)

    return rotated_img
        



def divide_columns(image:Union[str,cv2.typing.MatLike],method:str='WhittakerSmoother',logs:bool=False)->list[Box]:
    '''Get areas of columns based on black pixel frequency.\n
    Frequencies are then inverted to find white peaks.
    Frequency graph is smoothened using chosen method.
    
    Available methods:
        - WhittakerSmoother
        - savgol_filter'''
    columns = []

    methods = ['WhittakerSmoother','savgol_filter']
    if method not in methods:
        method = 'WhittakerSmoother'

    if isinstance(image,str):
        image = cv2.imread(image)

    original_height = image.shape[0] # height of original image (for columns dimensions)


    # auto crop margins
    image_crop = cut_document_margins(image=image,method='WhittakerSmoother')
    image = image[image_crop.top:image_crop.bottom,image_crop.left:image_crop.right]
    # cut possible header and footer (cut 30% from top and 10% from bottom)
    image = image[round(image.shape[0]*0.3):image.shape[0]-round(image.shape[0]*0.1),:]

    # binarize
    binarized = binarize_fax(image,logs=logs)

    # get frequency of white pixels per x axis
    x_axis_freq = np.zeros(binarized.shape[1])

    ## count when pixel neighbours (above,bellow and right) are also white
    mask = np.logical_and(
        np.logical_and(binarized[1:,:-1] == 255, binarized[:-1,:-1] == 255),
        binarized[:-1,1:] == 255
    )
    x_axis_freq = np.add.reduce(mask, axis=0)


    if x_axis_freq.any():
        # invert frequencies
        max_freq = max(x_axis_freq)
        x_axis_freq = np.array([max_freq - i for i in x_axis_freq])

        # smoothen frequencies
        if method == 'WhittakerSmoother':
            whittaker_smoother = WhittakerSmoother(lmbda=2e4, order=2, data_length = len(x_axis_freq))
            x_axis_freq_smooth = whittaker_smoother.smooth(x_axis_freq)
        elif method == 'savgol_filter':
            x_axis_freq_smooth = savgol_filter(x_axis_freq, round(len(x_axis_freq)*0.1), 2)

        x_axis_freq_smooth = [i if i > 0 else 0 for i in x_axis_freq_smooth ]



        peaks,_ = find_peaks(x_axis_freq_smooth,prominence=0.2*(max(x_axis_freq_smooth)- min(x_axis_freq_smooth)))

        x_axis_freq_smooth = np.array(x_axis_freq_smooth)

        # average of frequency
        average_smooth_frequency = np.average(x_axis_freq_smooth)

        if logs:
            
            # create 4 plots
            plt.subplot(2, 2, 1)
            plt.plot(peaks, x_axis_freq[peaks], "ob"); plt.plot(x_axis_freq); plt.legend(['prominence'])
            plt.title('Frequency')


            plt.subplot(2, 2, 2)
            plt.plot(peaks, x_axis_freq_smooth[peaks], "ob"); plt.plot(x_axis_freq_smooth); plt.legend(['prominence'])
            # average line
            plt.plot([0,len(x_axis_freq_smooth)], [average_smooth_frequency, average_smooth_frequency], "r--");
            plt.title('Frequency Smooth')

            # binarized image
            plt.subplot(2, 2, 3)
            plt.imshow(binarized, cmap='gray')
            plt.title('Binarized Image')

            plt.show()

        if logs:
            print('Peaks',peaks)

        # estimate columns
        potential_columns = []
        next_column = [0,None]
        for i in range(len(peaks)):

            if next_column[0] != None:
                next_column[1] = peaks[i]

            if next_column[0] != None and next_column[1] != None:
                potential_columns.append([next_column[0],next_column[1]])

            next_column = [peaks[i],None]

        # last column, until right margin
        if next_column[0] != None:
            next_column[1] = len(x_axis_freq_smooth)
            potential_columns.append(next_column)

        fix_pad = image_crop.left
        # create columns
        if potential_columns:
            if logs:
                print('potential columns',potential_columns)
            for column in potential_columns:
                c = Box({'left':int(column[0] + fix_pad),'right':int(column[1] + fix_pad),'top':0,'bottom':int(original_height)})
                columns.append(c)
        

    return columns



def descend_peak(signal:list, peak:int, direction:str='right')->int:
    '''Descend peak of signal. Returns index of last point in descent'''
    to_right = None
    lowest_point = peak

    if direction == 'right':
        if peak <= len(signal) - 1:
            if signal[peak] >= signal[peak + 1]:
                to_right = True
            elif peak > 0:
                if signal[peak] >= signal[peak - 1]:
                    to_right = False

    elif direction == 'left':
        if peak > 0:
            if signal[peak] >= signal[peak - 1]:
                to_right = False
            elif peak <= len(signal) - 1:
                if signal[peak] >= signal[peak + 1]:
                    to_right = True

    if to_right != None:
        if to_right:
            while lowest_point < len(signal) - 1:
                if signal[lowest_point] < signal[lowest_point + 1]:
                    break
                lowest_point = lowest_point + 1
        else:
            while lowest_point > 0:
                if signal[lowest_point] < signal[lowest_point - 1]:
                    break
                lowest_point = lowest_point - 1

    return lowest_point


def cut_document_margins(image:Union[str,cv2.typing.MatLike], method:str='WhittakerSmoother', logs:bool=False)->Box:
    '''
    Cut document margins by analysing pixel frequency.
    
    Margins should be great peaks of black pixels followed or preceded (depending on the side) by a great drop in frequency.'''

    cut_document = None

    methods = ['WhittakerSmoother','savgol_filter']
    if method not in methods:
        method = 'WhittakerSmoother'

    if type(image) == str:
        if not os.path.exists(image):
            print('Image not found')
            return cut_document

        image = cv2.imread(image)

    if logs:
        print('Image shape',image.shape)


    cut_document = Box({'left':0,'right':image.shape[1],'top':0,'bottom':image.shape[0]})

    binarized = binarize_fax(image,invert=True,logs=logs)

    # get frequency of black pixels per column
    x_axis_freq = np.add.reduce(binarized, axis=0)

    if x_axis_freq.any():
        # add 5% of length before and after
        x_axis_freq = np.append(np.zeros(int(len(x_axis_freq)*0.05)),x_axis_freq)
        x_axis_freq = np.append(x_axis_freq,np.zeros(int(len(x_axis_freq)*0.05)))

        pad = len(x_axis_freq) - binarized.shape[1]
        if logs:
            print('pad',pad)

        if method == 'WhittakerSmoother':
            whittaker_smoother = WhittakerSmoother(lmbda=2e4, order=2, data_length = len(x_axis_freq))
            x_axis_freq_smooth = whittaker_smoother.smooth(x_axis_freq)
        elif method == 'savgol_filter':
            x_axis_freq_smooth = savgol_filter(x_axis_freq, round(len(x_axis_freq)*0.1), 2)

        x_axis_freq_smooth = [i if i > 0 else 0 for i in x_axis_freq_smooth ]


        peaks,_ = find_peaks(x_axis_freq_smooth,prominence=0.2*(max(x_axis_freq_smooth)- min(x_axis_freq_smooth)))
        

        x_axis_freq_smooth = np.array(x_axis_freq_smooth)

        # average of frequency
        average_smooth_frequency = np.average(x_axis_freq_smooth)

        if logs:
            
            # create 4 plots
            plt.subplot(2, 2, 1)
            plt.plot(peaks, x_axis_freq[peaks], "ob"); plt.plot(x_axis_freq); plt.legend(['prominence'])
            plt.title('Frequency')


            plt.subplot(2, 2, 2)
            plt.plot(peaks, x_axis_freq_smooth[peaks], "ob"); plt.plot(x_axis_freq_smooth); plt.legend(['prominence'])
            # average line
            plt.plot([0,len(x_axis_freq_smooth)], [average_smooth_frequency, average_smooth_frequency], "r--");
            plt.title('Frequency Smooth')

            # binarized image
            plt.subplot(2, 2, 3)
            plt.imshow(binarized, cmap='gray')
            plt.title('Binarized Image')

            plt.show()

        if logs:
            print('Peaks',peaks)


        if peaks.any():

            # get left and right margins potential peaks
                # they need to be between out of the 10% to 90% range of the x axis of the image (20% and 80% respectively in the padded frequency array)
            left_margin = peaks[0] if peaks[0] <= len(x_axis_freq_smooth)*0.2 else None
            right_margin = None if len(peaks) == 1 else peaks[-1] if peaks[-1] >= len(x_axis_freq_smooth)*0.8 else None

            if logs:
                print('left margin',left_margin)
                print('right margin',right_margin)

            max_freq = max(x_axis_freq_smooth)
            # check left margin
            if left_margin:
                valid = False

                # peak needs to be followed by a drop to less than 10% of max frequency
                last_point = descend_peak(x_axis_freq_smooth,left_margin,'right')

                if logs:
                    print('Left - lowest point:',last_point)
                    print(x_axis_freq_smooth[last_point])

                if x_axis_freq_smooth[last_point] <= max_freq*0.1:
                    left_margin = last_point
                    valid = True

                if not valid:
                    last_point = descend_peak(x_axis_freq_smooth,left_margin,'left')

                    if x_axis_freq_smooth[last_point] <= max_freq*0.1:
                        left_margin = last_point
                    else:
                        left_margin = 0

            else: 
                left_margin = 0

            # check right margin
            if right_margin:
                valid = False

                # peak needs to be preceded by a drop to less than 10% of max frequency
                last_point = descend_peak(x_axis_freq_smooth,right_margin,'left')

                if logs:
                    print('Right - lowest point:',last_point)
                    print(x_axis_freq_smooth[last_point])

                if x_axis_freq_smooth[last_point] <= max_freq*0.1:
                    right_margin = last_point

                if not valid:
                    last_point = descend_peak(x_axis_freq_smooth,right_margin,'right')

                    if x_axis_freq_smooth[last_point] <= max_freq*0.1:
                        right_margin = last_point
                    else:
                        right_margin = len(x_axis_freq_smooth)

            else: 
                right_margin = len(x_axis_freq_smooth)

            # adjust margins to remove padding
            ## remove pad from left side
            if left_margin != 0:
                left_margin = int(abs(pad/2-left_margin))

            ## remove pad from right side if right margin is within right padded zone
            if right_margin > len(x_axis_freq_smooth) - pad/2:
                right_margin -= right_margin-(len(x_axis_freq_smooth) - pad/2)
            
            right_margin = int(right_margin-pad/2)


            if logs:
                print('Left margin',left_margin)
                print('Right margin',right_margin)

                # draw plot with left and right margins
                plt.imshow(binarized, cmap='gray')

                if left_margin:
                    plt.plot([left_margin,left_margin], [0,binarized.shape[0]], "r--");
                if right_margin:
                    plt.plot([right_margin,right_margin], [0,binarized.shape[0]], "r--");
                

                plt.show()


            cut_document = Box(left_margin,right_margin,0,binarized.shape[0])

    return cut_document


def binarize(image:Union[str,cv2.typing.MatLike],denoise_strength:int=10,invert:bool=False,logs:bool=False)->np.ndarray:
    '''Binarize image to black and white. 

    Parameters
    ----------
    image : Union[str,cv2.typing.MatLike]
        Image to binarize
    denoise_strength : int, optional
        Strength of denoise, by default 10. If 'auto', calculates SNR of image and chooses the best denoise strength (WIP).
    logs : bool, optional
        Print logs, by default False'''

    if isinstance(image,str):
        image = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    else:
        # check if image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    # determine denoise strength
    ## calculates SNR of image and chooses the best denoise strength
    if denoise_strength == 'auto':
        image_std = np.std(image)
        image_mean = np.mean(image)
        image_snr = image_mean/image_std
        denoise_strength = int(image_snr)
    elif denoise_strength is None:
        denoise_strength = 10

    if logs:
        print('Auto denoise strength:',denoise_strength)

    # denoise
    image = cv2.fastNlMeansDenoising(image,None,denoise_strength,7,21)

    # binarize
    type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    if invert:
        type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    image = cv2.threshold(image, 0, 255,type)[1]
    return image


def level_image(image:Union[str,cv2.typing.MatLike], black_point:Union[int,float]=0, white_point:Union[int,float]=255, gamma:float=1.0, is_percentage:bool=False)->np.ndarray:
    '''Level image to black and white.'''
    if isinstance(image,str):
        image = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    else:
        # check if image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert black_point and white_point from percentages to absolute values if needed
    if is_percentage:
        black_point = (black_point / 100.0) * 255
        white_point = (white_point / 100.0) * 255
    
    # Ensure the points are within valid range
    black_point = max(0, min(255, black_point))
    white_point = max(0, min(255, white_point))
    
    # Stretch or compress the contrast
    scale = 255.0 / (white_point - black_point)
    image = np.clip((image - black_point) * scale, 0, 255).astype(np.uint8)
    
    # Apply gamma correction
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, table)
    
    return image


def binarize_fax(image:Union[str,cv2.typing.MatLike],g_kernel_size:int=30,g_sigma:int=15,black_point:Union[int,float]=10,
                 white_point:Union[int,float]=90,gamma:float=0.2,is_percentage:bool=True,invert:bool=False,logs:bool=False)->np.ndarray:
    '''Binarize image using fax binarization algorithm.
    
    Algorithm:
    convert "image" -colorspace Gray ( +clone -blur 15,15 ) -compose Divide_Src -composite -level 10%,90%,0.2
    '''
    if isinstance(image,str):
        image = cv2.imread(image)

    # step 1 - convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply a blur (15,15)
    # blurred = cv2.blur(gray,(30,30)) # closest results
    kernel = cv2.getGaussianKernel(g_kernel_size, g_sigma)
    blurred = cv2.sepFilter2D(gray, -1, kernel, kernel)

    # Step 3: Composite operation (Divide_Src)
    composite = cv2.divide(gray,blurred, scale=255)

    # Step 4: Adjust levels (emulate -level 10%,90%,0.2)
    level = level_image(composite,black_point,white_point,gamma,is_percentage=is_percentage)

    if invert:
        level = 255 - level

    return level



def canny_edge_detection(image:cv2.typing.MatLike): 
    # Convert the frame to grayscale for edge detection 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
      
    # Apply Gaussian blur to reduce noise and smoothen edges 
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5) 
      
    # Perform Canny edge detection 
    edges = cv2.Canny(blurred, 70, 135) 
      
    return blurred, edges

def draw_countours(image:Union[str,cv2.typing.MatLike]):
    '''Draw contours on image'''
    if isinstance(image,str):
        image = cv2.imread(image)

    _, edges = canny_edge_detection(image)

    return edges


def identify_document_images(image:Union[str,cv2.typing.MatLike],tmp_dir:str=None,logs:bool=False)->list[Box]:
    '''Identify document images in image. Uses leptonica's page segmentation function to identify document images.

    Parameters
    ----------
    image : Union[str,cv2.typing.MatLike]
        Image to identify document images
    logs : bool, optional
        Print logs, by default False
    Returns
    -------
    List[Box]
        List of boxes with document images'''
    
    if tmp_dir is None:
        tmp_dir = f'{file_path}/tmp'

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # binarize image and save to 1bpp format
    binarized = binarize(image,logs=logs)

    tmp_file = f'{tmp_dir}/tmp_binarized.png'
    cv2.imwrite(tmp_file,binarized,params=[cv2.IMWRITE_PNG_BILEVEL, 1])

    # run leptonica script
    if logs:
        print('Running page segmentation using leptonica.')

    os.system(f'{file_path}/leptonica_lib/segment_doc {tmp_file} {tmp_dir}')

    # read leptonica output
    leptonica_output_path = f'{tmp_dir}'
    image_boxes_output_path = f'{leptonica_output_path}/htmask.boxa'
    if not os.path.exists(image_boxes_output_path):
        return []
    
    image_boxes = open(image_boxes_output_path,'r',encoding='utf-8').readlines()

    # parse leptonica output
    boxes = []
    for line in image_boxes:
        box_pattern = r'Box\[\d+\]:\s+x = (\d+),\s+y = (\d+),\s+w = (\d+),\s+h = (\d+)'
        if not re.search(box_pattern,line):
            continue

        match = re.search(box_pattern,line)
        x = int(match.group(1))
        y = int(match.group(2))
        w = int(match.group(3))
        h = int(match.group(4))

        boxes.append(Box(x,x+w,y,y+h))

    # remove tmp files
    os.remove(tmp_file)

    return boxes

def remove_document_images(image:Union[str,cv2.typing.MatLike],doc_images:list[Box]=None,tmp_dir:str=None,logs:bool=False):
    '''Remove document images in image'''

    if isinstance(image,str):
        image = cv2.imread(image)

    if tmp_dir is None:
        tmp_dir = f'{file_path}/tmp'

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if not doc_images:
        # identify document images
        doc_images = identify_document_images(image.copy(),tmp_dir=tmp_dir,logs=logs)

    average_color = [int(np.average(image[:,:,i])) for i in range(3)]
    # remove document images
    for box in doc_images:
        x = box.left
        y = box.top
        w = box.width
        h = box.height

        image[y:y+h,x:x+w] = average_color

    return image


def segment_document_elements(image:Union[str,cv2.typing.MatLike],tmp_dir:str=None,logs:bool=False)->tuple[list[Box],list[Box]]:
    '''Segment document into text and image bounding boxes using leptonica.
    
    Outputs
    -------
    list[Box]
        List of text bounding boxes
    list[Box]
        List of image bounding boxes
    '''
    if isinstance(image,str):
        image = cv2.imread(image)

    images_bb = []
    text_bbs = []

    if tmp_dir is None:
        tmp_dir = f'{file_path}/tmp'

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # binarize image and save to 1bpp format
    binarized = binarize(image,logs=logs)

    tmp_file = f'{tmp_dir}/tmp_binarized.png'
    cv2.imwrite(tmp_file,binarized,params=[cv2.IMWRITE_PNG_BILEVEL, 1])

    # run leptonica script
    if logs:
        print('Running page segmentation using leptonica.')

    os.system(f'{file_path}/leptonica_lib/segment_doc {tmp_file} {tmp_dir}')

    # read leptonica output
    leptonica_output_path = f'{tmp_dir}'
    image_boxes_output_path = f'{leptonica_output_path}/htmask.boxa'
    text_boxes_output_path = f'{leptonica_output_path}/textmask.boxa'

    # read image boxes output
    if os.path.exists(image_boxes_output_path):
        image_boxes = open(image_boxes_output_path,'r',encoding='utf-8').readlines()

        # parse leptonica output
        
        for line in image_boxes:
            box_pattern = r'Box\[\d+\]:\s+x = (\d+),\s+y = (\d+),\s+w = (\d+),\s+h = (\d+)'
            if not re.search(box_pattern,line):
                continue

            match = re.search(box_pattern,line)
            x = int(match.group(1))
            y = int(match.group(2))
            w = int(match.group(3))
            h = int(match.group(4))

            images_bb.append(Box(x,x+w,y,y+h))

    # read text boxes output
    if os.path.exists(text_boxes_output_path):
        text_boxes = open(text_boxes_output_path,'r',encoding='utf-8').readlines()

        # parse leptonica output
        for line in text_boxes:
            box_pattern = r'Box\[\d+\]:\s+x = (\d+),\s+y = (\d+),\s+w = (\d+),\s+h = (\d+)'
            if not re.search(box_pattern,line):
                continue

            match = re.search(box_pattern,line)
            x = int(match.group(1))
            y = int(match.group(2))
            w = int(match.group(3))
            h = int(match.group(4))

            text_bbs.append(Box(x,x+w,y,y+h))

    return text_bbs,images_bb

    
def clean_delimiters_connected_component(delimiters:list[Box],image:Union[str,cv2.typing.MatLike],max_components:int=3,logs:bool=False,debug:bool=False)->list[Box]:
    '''Try to reduce the number of delimiters by removing potential delimiters with too many connected components (probably noise or text).'''

    if logs:
        print('Cleaning delimiters.')

    if isinstance(image,str):
        image = cv2.imread(image)


    # remove low confidence delimiters
    ## to many conected componentes (probably noise or text)
    i = 0
    removed = 0
    while i < len(delimiters):
        delimiter = delimiters[i]
        left = delimiter.left
        right = delimiter.right
        top = delimiter.top
        bottom = delimiter.bottom
        # add some padding
        if left == right :
            right += 1
        if top == bottom:
            bottom += 1

        image_portion = image[top:bottom,left:right]
        componentes = cv2.connectedComponents(image_portion, 8, cv2.CV_32S)
        n_components = componentes[0]
        # remove delimiters with too many connected components
        if n_components > max_components:
            if debug:
                print(f'Removing delimiter {delimiter.id} with {componentes[0]} connected components.')
            delimiters.pop(i)
            removed += 1
            i -= 1

        i += 1

    if logs:
        print(f'Conected Components: Removed {removed} delimiters. Remaining {len(delimiters)} delimiters.')

    return delimiters

def clean_delimiters_unite(delimiters:list[Box],image:Union[str,cv2.typing.MatLike],id:bool=True,logs:bool=False,debug:bool=False)->list[Box]:
    '''Try to reduce the number of delimiters by uning delimiters that are close to each other.
    
    * Unite delimiters that are close to each other (intercepting or less than 1% of total image width or height)
    * Same direction
    * Within respective borders'''

    if isinstance(image,str):
        image = cv2.imread(image)

    # id delimiters
    if id:
        i = 0
        while i < len(delimiters):
            delimiters[i].id = i
            i += 1

    intersects = False
    # unite delimiters that are close to each other (intercepting or less than 5% of total image width or height)
    ## same direction
    ## within respective borders
    i = 0
    o_delimiters = len(delimiters)
    range_x = int(image.shape[1]*0.005)
    range_y = int(image.shape[0]*0.005)
    while i < len(delimiters):
        delimiter = delimiters[i]
        orientation = delimiter.get_box_orientation()
        j = 0
        # compare delimiter with every other
        while j < len(delimiters):
            compare_delimiter = delimiters[j]

            if delimiter.id == compare_delimiter.id:
                j += 1
                continue

            compare_delimiter_orientation = compare_delimiter.get_box_orientation()
            # check if same direction
            if compare_delimiter_orientation == orientation:
                # range_x = range_x_h if orientation == 'horizontal' else range_x_v
                # range_y = range_y_v if orientation == 'vertical' else range_y_h
                distance = compare_delimiter.distance_to(delimiter,border='closest',range_x=range_x,range_y=range_y,range_type='absolute')
                # check if close or intersect and within borders
                ## if join, restart loop
                if ((distance < image.shape[1]*0.005 and orientation == 'horizontal') or (distance < image.shape[1]*0.01 and orientation == 'vertical') or (intersects:=delimiter.intersects_box(compare_delimiter,inside=True)))\
                      and delimiter.within_horizontal_boxes(compare_delimiter,range=range_x,range_type='absolute'):
                    if debug:
                        print(f'Joining vertically delimiters {delimiter.id} and {compare_delimiter.id} | distance: {distance} | orientation: {orientation} | intersects: {intersects}')

                    delimiter.join(compare_delimiter)
                    delimiters[i] = delimiter
                    delimiters.pop(j)
                    i = i - 1 if j < i else i
                    j = -1
                elif ((distance < image.shape[0]*0.01 and orientation == 'horizontal') or (distance < image.shape[0]*0.005 and orientation == 'vertical') or (intersects:=delimiter.intersects_box(compare_delimiter,inside=True)))\
                      and delimiter.within_vertical_boxes(compare_delimiter,range=range_y,range_type='absolute'):
                    if debug:
                        print(f'Joining horizontally delimiters {delimiter.id} and {compare_delimiter.id} | distance: {distance} | orientation: {orientation} | intersects: {intersects}')

                    delimiter.join(compare_delimiter)
                    delimiters[i] = delimiter
                    delimiters.pop(j)
                    i = i - 1 if j < i else i
                    j = -1
            j += 1
        i += 1


    if logs:
        print(f'Unite delimiters: Removed {o_delimiters - len(delimiters)} delimiters. Remaining {len(delimiters)} delimiters.')

    return delimiters



def clean_delimiters_intersections(delimiters:list[Box],image:Union[str,cv2.typing.MatLike],id:bool=True,logs:bool=False,debug:bool=False)->list[Box]:
    '''Try to reduce the number of delimiters by removing potential delimiters with too many intersections between differently oriented delimiters.'''

    if logs:
        print('Cleaning delimiters.')

    if isinstance(image,str):
        image = cv2.imread(image)

    # id delimiters
    if id:
        i = 0
        while i < len(delimiters):
            delimiters[i].id = i
            i += 1

    # compare delimiter with every other
    ## if intersects with delimiter with different orientation, remove
    i = 0
    removed = 0
    while i < len(delimiters):
        delimiter = delimiters[i]
        orientation = delimiter.get_box_orientation()
        j = i
        while j < len(delimiters):
            compare_delimiter = delimiters[j]
            compare_delimiter_orientation = compare_delimiter.get_box_orientation()
            if delimiter.id == compare_delimiter.id:
                j += 1
                continue
            if orientation != compare_delimiter_orientation and delimiter.intersects_box(compare_delimiter):
                if debug:
                    print(f'Removing delimiter {delimiter.id} with {compare_delimiter.id} - intersections.')
                delimiters.pop(i)
                removed += 1
                i -= 1
                break
            j += 1
        i += 1

    if logs:
        print(f'Intersections: Removed {removed} delimiters. Remaining {len(delimiters)} delimiters.')

    return delimiters



def clean_delimiters(delimiters:list[Box],image:Union[str,cv2.typing.MatLike],check_connected_components:bool=True,unite_delimiters:bool=True,check_intersections:bool=True,logs:bool=False,debug:bool=False)->list[Box]:
    '''Try to reduce the number of delimiters by uning delimiters that are close to each other and removing low confidence delimiters (multiple intersections and low joint component).'''

    if logs:
        print('Cleaning delimiters.')

    if isinstance(image,str):
        image = cv2.imread(image)

    # apply transformation to unite delimiters that are close to each other
    dilated=cv2.morphologyEx(image, cv2.MORPH_DILATE, (3,3))
    tresh = cv2.threshold(dilated, 128, 255, cv2.THRESH_BINARY)[1]

    # id all delimiters
    i = 0
    for delimiter in delimiters:
        delimiter.id = i
        i += 1
    if debug:
        show = draw_bounding_boxes(image,delimiters,id=True)
        show = cv2.resize(show,(1000,1200))
        cv2.imshow('image',show)
        cv2.waitKey(0)



    if check_intersections:
        delimiters = clean_delimiters_intersections(delimiters,image,id=False,logs=logs,debug=debug)

    if unite_delimiters:
        delimiters = clean_delimiters_unite(delimiters,image,id=False,logs=logs,debug=debug)

    if check_connected_components:
        delimiters = clean_delimiters_connected_component(delimiters,tresh,max_components=20,logs=logs,debug=debug)

    return delimiters


def get_document_delimiters(image:Union[str,cv2.typing.MatLike],tmp_dir:str=None,min_length_h:int=None,min_length_v:int=None,max_line_gap_v:int=None,max_line_gap_h:int=None,reduce_delimiters:bool=True,logs:bool=False,debug:bool=False)->list[Box]:
    '''Get document delimiters in image using Hough lines. 

    reduce_delimiters option will apply clean_delimiters method.'''

    if not tmp_dir:
        tmp_dir = f'{file_path}/tmp'

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if isinstance(image,str):
        image = cv2.imread(image)

    # delimiter identification parameters
    if not min_length_h:
        min_length_h = int(image.shape[1]*0.01)

    if not min_length_v:
        min_length_v = int(image.shape[0]*0.01)

    if not max_line_gap_v:
        max_line_gap_v = int(image.shape[0]*0.01)

    if not max_line_gap_h:
        max_line_gap_h = int(image.shape[1]*0.01)


    # binarize image
    binarized = binarize_fax(image,logs=logs)

    # dilate
    morph_base = cv2.erode(binarized,(3,3),iterations = 1)

    ## horizontal lines
    ### identify, remove non horizontal and accentuate horizontal lines
    morph = cv2.erode(morph_base,(1,2),iterations = 1)
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(int(min_length_h),3))
    morph = cv2.dilate(morph,horizontal_structure,iterations = 1)
    morph = cv2.erode(morph,horizontal_structure,iterations = 1)

    ### get edges
    edges = cv2.Canny(morph,50,200,None,3)
    edges = cv2.dilate(edges,(2,4),iterations = 1)

    ### get hough lines
    horizontal_lines = cv2.HoughLinesP(edges,1,np.pi/180,50,None,minLineLength=min_length_h,maxLineGap=max_line_gap_h)

    ## vertical lines
    ### identify, remove non vertical and accentuate vertical lines
    morph = cv2.erode(morph_base,(2,1),iterations = 1)
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(1,int(min_length_v)))
    morph = cv2.dilate(morph,vertical_structure,iterations = 1)
    morph = cv2.erode(morph,vertical_structure,iterations = 1)

    ### get edges
    edges = cv2.Canny(morph,50,200,None,3)
    edges = cv2.dilate(edges,(4,2),iterations = 1)

    ### get hough lines
    vertical_lines = cv2.HoughLinesP(edges,1,np.pi/180,50,None,minLineLength=min_length_v,maxLineGap=max_line_gap_v)

    lines = []

    # merge lines
    if horizontal_lines is not None and vertical_lines is not None:
        lines = np.concatenate((horizontal_lines,vertical_lines),axis=0)
    elif horizontal_lines is not None:
        lines = horizontal_lines
    elif vertical_lines is not None:
        lines = vertical_lines

    # get delimiters
    delimiters = []
    for line in lines:
        l = line[0]

        x0 = l[0]
        y0 = l[1]
        x1 = l[2]
        y1 = l[3]

        t5 = math.tan(5*math.pi/180)
        t60 = math.sqrt(3)/2
        dx = x1 - x0
        dy = y1 - y0
        is_vertical = dy != 0 and abs(dx/dy) < t5
        is_horizontal = dx != 0 and abs(dy/dx) < t60
        # if not horizontal and not vertical, skip
        if not is_vertical and not is_horizontal:
            if debug:
                print(f'Line {l} is not horizontal or vertical.')
            continue
        
        left = min(x0,x1)
        right = max(x0,x1)
        top = min(y0,y1)
        bottom = max(y0,y1)

        delimiter = Box(left,right,top,bottom)
        delimiters.append(delimiter)

    # remove potential border delimiters
    for delimiter in delimiters:
        if delimiter.left in [0,image.shape[1]] or\
            delimiter.top in [0,image.shape[0]] or\
            delimiter.right in [0,image.shape[1]] or\
            delimiter.bottom in [0,image.shape[0]]:
            if debug:
                print(f'Removing delimiter {delimiter.id} because it is in the border.')
            delimiters.remove(delimiter)


    # clean delimiters
    if reduce_delimiters:
        delimiters = clean_delimiters(delimiters,binarized,check_connected_components=False,logs=logs,debug=debug)


    # check if delimiters have a minimum size (5% of image size)
    # also remove delimiters in border
    minimum_size_h = image.shape[1]*0.05
    minimum_size_v = image.shape[0]*0.05
    i = 0
    while i < len(delimiters):
        delimiter = delimiters[i]
        orientation = delimiter.get_box_orientation()
        if orientation == 'horizontal':
            if delimiter.width < minimum_size_h or \
                delimiter.top == 0 or delimiter.bottom == image.shape[0]:
                if debug:
                    print(f'Removing delimiter {delimiter.id} because it is too small.')
                delimiters.remove(delimiter)
                i -= 1
        elif orientation == 'vertical':
            if delimiter.height < minimum_size_v or \
                delimiter.left == 0 or delimiter.right == image.shape[1]:
                if debug:
                    print(f'Removing delimiter {delimiter.id} because it is too small.')
                
                delimiters.remove(delimiter)
                i -= 1

        i += 1

    if logs:
        print(f'Found {len(delimiters)} delimiters.')


    if debug:
        # id all delimiters
        i = 0
        for delimiter in delimiters:
            delimiter.id = i
            i += 1
        show = draw_bounding_boxes(image.copy(),delimiters,id=True)
        show = cv2.resize(show,(1000,1200))
        cv2.imshow('image',show)
        cv2.waitKey(0)
    
    return delimiters




def segment_document_delimiters(image:Union[str,cv2.typing.MatLike],delimiters:list[Box],target_segments:list[str]=['header','body','footer'],logs:bool=False,debug:bool=False)->list[Box]:
    '''Segment document into header, body and footer using list of delimiters'''

    header = None
    body = None
    footer = None

    if isinstance(image,str):
        image = cv2.imread(image)

    # find header delimiter
    ## has to be in the upper 30% of the image, horizontal and ith a lenght of at least 40% of the image
    potential_header_delimiters = []
    for delimiter in delimiters:
        
        if delimiter.get_box_orientation() == 'horizontal' and delimiter.bottom <= 0.3*image.shape[0] and delimiter.width >= 0.4*image.shape[1]:
            potential_header_delimiters.append(delimiter)

    if len(potential_header_delimiters) > 0:
        # sort delimiters by y position (lower is better)
        potential_header_delimiters.sort(key=lambda x: x.top)
        header = potential_header_delimiters[-1]


    # find footer delimiter
    ## has to be in the lower 30% of the image, horizontal and ith a lenght of at least 40% of the image
    potential_footer_delimiters = []
    for delimiter in delimiters:
        
        if delimiter.get_box_orientation() == 'horizontal' and delimiter.top >= 0.7*image.shape[0] and delimiter.width >= 0.4*image.shape[1]:
            potential_footer_delimiters.append(delimiter)

    if len(potential_footer_delimiters) > 0:
        # sort delimiters by y position (higher is better)
        potential_footer_delimiters.sort(key=lambda x: x.top,reverse=True)
        footer = potential_footer_delimiters[0]

    # create bboxes for header, body and footer
    ## default body (whole image)
    body = Box(0,int(image.shape[1]),0,int(image.shape[0]))
    if header is not None:
        header = Box(0,int(image.shape[1]),0,header.bottom)
    else:
        header = Box(0,0,0,0)

    if footer is not None:
        footer = Box(0,int(image.shape[1]),footer.top,int(image.shape[0]))
    else:
        footer = Box(0,0,0,0)

    ## remove heade and footer from body
    if 'header' in target_segments:
        body.remove_box_area(header)
    else:
        header = Box(0,0,0,0)

    if 'footer' in target_segments:
        body.remove_box_area(footer)
    else:
        footer = Box(0,0,0,0)


    return [header,body,footer]



def segment_document(image:Union[str,cv2.typing.MatLike],remove_images:bool=True,tmp_dir:str=None,target_segments:list[str]=['header','body','footer'],logs:bool=False,debug:bool=False)->tuple[Box,Box,Box]:
    '''Segment document into header, body and footer using delimiters. Uses aux function: segment_document_delimiters.'''
    
    if not tmp_dir:
        tmp_dir = f'{file_path}/tmp'

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if isinstance(image,str):
        image = cv2.imread(image)

    if remove_images:
        image = remove_document_images(image,tmp_dir=tmp_dir,logs=logs)

    delimiters = get_document_delimiters(image,tmp_dir=tmp_dir,logs=logs,debug=debug)

    header,body,footer = segment_document_delimiters(image,delimiters,target_segments=target_segments,logs=logs,debug=debug)

    return header,body,footer




def draw_bounding_boxes(image:Union[str,cv2.typing.MatLike],boxes:list[Box],color:tuple=(0,255,0),custom_color:bool=False,id:bool=False,logs:bool=False,debug:bool=False)->cv2.typing.MatLike:
    '''Draw bounding boxes on image'''
    if isinstance(image,str):
        image = cv2.imread(image)

    for box in boxes:
        box_color = color
        if custom_color:
            try:
                box_color = box.color
            except:
                pass
        cv2.rectangle(image,(box.left,box.top),(box.right,box.bottom),box_color,2)

        if id:
            cv2.putText(image,str(box.id),(box.left,box.top),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

    return image