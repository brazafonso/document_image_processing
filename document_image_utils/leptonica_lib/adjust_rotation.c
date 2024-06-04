#include <leptonica/allheaders.h>

int main(int argc, char *argv[]) {

  if (argc != 3) {
    fprintf(stderr, "Usage: %s <image> <output>\n", argv[0]);
    return 1;
  }

  char *filename = argv[1];
  char *output = argv[2];

  PIX *pix2;
  l_float32 angle, conf;
  PIX *image = pixRead(filename);
  pix2 = pixFindSkewAndDeskew(image, 2, &angle, &conf);
  // printf("Skew angle: %7.2f degrees; %6.2f conf\n", angle, conf);
  pixWrite(output, pix2, IFF_PNG);

  pixDestroy(&image);
  pixDestroy(&pix2);
  return 0;
}