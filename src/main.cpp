#include <getopt.h>

#include <cassert>
#include <iostream>
#include <stdio.h>

#include "CycleTimer.h"
#include "loadDataset.h"
#include "model.h"

using namespace std;

// training and testing dataset size
int train_size = 40960;
const int test_size = 10000;

int main(int argc, char **argv) {
  char opt;
  static struct option long_options[] = {{"batch-size", 1, nullptr, 'b'},
                                         {"hidden-size", 1, nullptr, 'h'},
                                         {"train-size", 1, nullptr, 't'},
                                         {"epoch-count", 1, nullptr, 'e'},
                                         {0, 0, 0, 0}};
  size_t batch_size = 8192;
  int epoch = 3;

  while ((opt = getopt_long(argc, argv, "b:h:t:e:x:y:p:", long_options, nullptr)) !=
         EOF) {
    switch (opt) {
      case 'b': {
        size_t new_size = stoul(optarg);
        batch_size = new_size;
        break;
      }
      case 'h': {
        int new_size = stoi(optarg);
        set_hidden_layer_size(new_size);
        break;
      }
      case 't': {
        int new_size = stoi(optarg);
        train_size = new_size;
        break;
      }
      case 'e': {
        int new_size = stoi(optarg);
        epoch = new_size;
        break;
      }
      case 'x': {
        int new_x = stoi(optarg);
        set_partition_x(new_x);
      }
      case 'y': {
        int new_y = stoi(optarg);
        set_partition_y(new_y);
      }
      case 'p': {
        int part = stoi(optarg);
        set_partition(part);
      }
    }
  }

  // read training data
  int *x, *y;
  readTrainData(&x, &y);

  data_t *x_train = (data_t *)malloc(train_size * input_dim * sizeof(data_t));
  data_t *y_train = (data_t *)malloc(train_size * output_dim * sizeof(data_t));
  memset(y_train, 0, train_size * output_dim * sizeof(data_t));

  for (int i = 0; i < train_size; ++i) {
    for (int j = 0; j < input_dim; ++j) {
      x_train[i * input_dim + j] = (double)x[i * input_dim + j] / 255.0;
    }
    y_train[i * output_dim + y[i]] = 1.0;
  }

  int tmp;
  while (~scanf("%d", &tmp) && tmp != -1) printImage(tmp, x, y);

  // send to device
  x_train = toDevice(train_size * input_dim, x_train);
  y_train = toDevice(train_size * output_dim, y_train);
  y = toDevice(train_size, y);

  double start_time = CycleTimer::currentSeconds();
  model myModel;
  for (int i = 1; i <= epoch; ++i) {
    int iter = (train_size-1) / (batch_size)+1;
    data_t *pred, loss = 0, _accuracy_ = 0;

    // learning rate
    data_t lr = 0.1;
    // printf("lr: %f\n", lr);

    for (int j = 0; j < iter; ++j) {
      myModel.zero_grad();
      if(j == iter-1) {
        int tmp = train_size % batch_size;
        if (tmp == 0) tmp = batch_size;
        pred = myModel.forward(tmp, x_train + j * batch_size * input_dim);
        loss += myModel.loss(tmp, pred, y_train + j * batch_size * output_dim);
        _accuracy_ += accuracy(tmp, pred, y + j * batch_size);
      }
      else {
        pred = myModel.forward(batch_size, x_train + j * batch_size * input_dim);
        loss += myModel.loss(batch_size, pred, y_train + j * batch_size * output_dim);
        _accuracy_ += accuracy(batch_size, pred, y + j * batch_size);
      }
      myModel.backward();
      myModel.update(lr);
    }
    loss /= (data_t)iter;
    _accuracy_ /= (data_t)iter;
    printf("Epoch %d: Loss: %f, accuracy: %f\n", i, loss, _accuracy_);
  }
  double end_time = CycleTimer::currentSeconds();
  printf("time: %.4f sec\n", (end_time - start_time) / (double)epoch);
  // printf("%.4f", (end_time - start_time) / (double)epoch);
  // printf("test done.\n");

  return 0;
}