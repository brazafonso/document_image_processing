
import os
from document_image_utils.image import cut_document_margins


file_path = os.path.dirname(os.path.realpath(__file__))
study_cases_folder = f'{file_path}/../../../study_cases'

## TODO
# def test_cut_document_margins_1():
#     image = f'{study_cases_folder}/simple template/2-1.jpg'
#     assert cut_document_margins(image,logs=True) != None 

# def test_cut_document_margins_2():
#     image = f'{study_cases_folder}/simple template/2-2.jpg'
#     assert cut_document_margins(image,logs=True) != None 


