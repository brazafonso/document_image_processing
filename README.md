# Contextualization

In progress toolkit for document image pre processing.

Aimed for images to be OCRed.

# Main Available methods

- **Auto rotate image**

    Uses left margin of a document to calculate the angle of rotation present, and correct it accordingly.

    Can be given the rotation direction (clocwise or counter_clockwise), or in auto mode tries to determine the side to which the document is tilted (can be none, in which case image won't be rotated).

- **Calculate rotation direction**

    Calculates rotation direction of an image by finding the biggest sets of the first black pixels appearances (with outliers removed) in the image for each direction: clockwise, counter_clockwise and none.

    For none direction, the set is created based on pixels with same 'x' coordinate that with less than a 5% height difference, relative to the image's height.

- **Binarize document**

    Normal binarization with otsu tresholding and fastNlMeansDenoising.

    Fax binarization, following the image magick command: convert "image" -colorspace Gray ( +clone -blur 15,15 ) -compose Divide_Src -composite -level 10%,90%,0.2

- **Split document into columns**

    Analyzes document image pixel color frequency and split document image into columns.

- **Auto crop document**

    Analyzes document image pixel color frequency and cut document margins, aiming mostly to remove possible folds in the corners.

- **Identify document images**
    Identify document images in image, using algorithm available in leptonica's repository that finds potential image masks.

- **Get document delimiters**
    Get document delimiters, using image transformations.

- **Segment document**
    Segments document image into header, body and footer, using delimiters. Only the body is always guaranteed to have a value.

# Bash commands:

- **binarize** : binarize document image.

- **rotate_document** : rotate document image.

- **split_columns** : split document into column images.

- **d_auto_crop** : auto crop document image.