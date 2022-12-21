

#ifndef _PARALLEL_MLP_LOAD_DATASET_H
#define _PARALLEL_MLP_LOAD_DATASET_H

#include <stdio.h>
#include <stdlib.h>

inline unsigned char readchar(FILE *fp) {
  const int S = 1 << 20;
  static unsigned char buf[S], *p = buf, *q = buf;
  return p == q && (q = (p = buf) + fread(buf, 1, S, fp)) == buf ? EOF : *p++;
}

inline int readint(FILE *fp) {
  int ret = 0;
  char *c = (char *)&ret;
  for (int i = 3; i >= 0; --i) c[i] = readchar(fp);
  return ret;
}

void printImage(int idx, int *x, int *y) {
  int l = idx * 784;
  int r = l + 784;
  printf("label: %d\n", y[idx]);
  for (int i = l; i < r; ++i) {
    printf("%4d", x[i]);
    if (i % 28 == 27) printf("\n");
  }
}

void readTrainData(int **x, int **y) {
  // read image
  FILE *fx = fopen("dataset/train-images.idx3-ubyte", "rb");
  if (fx == NULL) {
    fprintf(stderr, "Can't open train data.\n");
    exit(-1);
  }
  if (readint(fx) != 2051) {
    fprintf(stderr, "Read the wrong file.\n");
    exit(-1);
  }
  int n = readint(fx);
  int r = readint(fx);
  int c = readint(fx);
  *x = (int *)malloc(n * r * c * sizeof(int));
  int *p = *x;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < r; ++j) {
      for (int k = 0; k < c; ++k) {
        *p = readchar(fx);
        ++p;
      }
    }
  }
  fclose(fx);
  // read label
  FILE *fy = fopen("dataset/train-labels.idx1-ubyte", "rb");
  if (readint(fy) != 2049) {
    fprintf(stderr, "Read the wrong file.\n");
    exit(-1);
  }
  n = readint(fy);
  p = *y = (int *)malloc(n * sizeof(int));
  for (int i = 0; i < n; ++i) {
    *p = readchar(fy);
    ++p;
  }
  printf("Load %d training data\n", n);
  fclose(fy);
}

void readTestData(int **x, int **y) {
  // read image
  FILE *fx = fopen("dataset/t10k-images.idx3-ubyte", "rb");
  if (fx == NULL) {
    fprintf(stderr, "Can't open train data.\n");
    exit(-1);
  }
  if (readint(fx) != 2051) {
    fprintf(stderr, "Read the wrong file.\n");
    exit(-1);
  }
  int n = readint(fx);
  int r = readint(fx);
  int c = readint(fx);
  *x = (int *)malloc(n * r * c * sizeof(int));
  int *p = *x;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < r; ++j) {
      for (int k = 0; k < c; ++k) {
        *p = readchar(fx);
        ++p;
      }
    }
  }
  fclose(fx);
  // read label
  FILE *fy = fopen("dataset/t10k-labels.idx1-ubyte", "rb");
  if (readint(fy) != 2049) {
    fprintf(stderr, "Read the wrong file.\n");
    exit(-1);
  }
  n = readint(fy);
  p = *y = (int *)malloc(n * sizeof(int));
  for (int i = 0; i < n; ++i) {
    *p = readchar(fy);
    ++p;
  }
  printf("Load %d testing data\n", n);
  fclose(fy);
}

#endif  // !_PARALLEL_MLP_LOAD_DATASET_H