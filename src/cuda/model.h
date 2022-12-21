#ifndef _MODEL_H_
#define _MODEL_H_

#include <math.h>
#include <string.h>

#include <random>

// model setting
const int input_dim = 784;
const int output_dim = 10;

void set_hidden_layer_size(int new_size);

// data type
using data_t = float;
void print(size_t n, data_t *a);
int *toDevice(const size_t n, int *x);
int *toHost(const size_t n, int *x);
data_t *toDevice(const size_t n, data_t *x);
data_t *toHost(const size_t n, data_t *x);

void fill_uniform(size_t n, data_t *a, const data_t L, const data_t R);
void fill_val(size_t n, data_t *a, const data_t val);

data_t sigmoid(data_t x);
data_t accuracy(const size_t batch_size, data_t *pred, int *label);

class model {
 private:
  // count the number of input batches
  data_t batch_count;
  // model parameters
  data_t *w_1, *w_2, *bias_1, *bias_2;
  // for gradient calculation
  data_t *grad_loss, *prev_grad;
  data_t *sum_1, *sum_2, *grad_sigmoid_1, *grad_sigmoid_2;
  data_t *grad_w_1, *grad_w_2, *grad_bias_1, *grad_bias_2;

 public:
  ~model();
  model();
  void backward();
  void zero_grad();
  void update(data_t lr);
  data_t *forward(size_t batch_size, data_t *x);
  data_t loss(size_t batch_size, data_t *pred, data_t *real);
};

#endif