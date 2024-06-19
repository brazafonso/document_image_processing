'''Module for document image processing.

Main available functions:\n

- binarize : binarize document image.

- rotate_document : rotate document image, with option to automatically try to rotate. May use leptonica for minor adjustments.

- split_columns : analyzes document image pixel color frequency and split document image into columns.

- cut_document_margins : analyzes document image pixel color frequency and cut document margins, aiming mostly to remove possible folds in the corners.

- identify_document_images : identify document images in image, using leptonica.

- get_document_delimiters : get document delimiters, using image transformations.

- segment_document : segments document image into header, body and footer, using delimiters.

Bash commands:\n

- binarize : binarize document image.

- rotate_document : rotate document image.

- split_columns : split document into column images.

- d_auto_crop : auto crop document image.
'''
