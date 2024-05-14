
import os
from document_image_utils.image import divide_columns


file_path = os.path.dirname(os.path.realpath(__file__))
study_cases_folder = f'{file_path}/../../../study_cases'


def test_divide_columns_1():
    image = f'{study_cases_folder}/simple template/2-1.jpg'
    assert len(divide_columns(image)) == 3

def test_divide_columns_2():
    image = f'{study_cases_folder}/simple template/2-2.jpg'
    assert len(divide_columns(image)) == 3


