#ifndef _MODEL_H_
#define _MODEL_H_

#include <string.h>
#include <math.h>
#include <random>

// model setting
const int input_dim = 784;
int hidden_dim = 300;
const int output_dim = 10;

void set_partition(int new_partition) {}
void set_partition_x(int new_x) {}
void set_partition_y(int new_y) {}

void set_hidden_layer_size(int new_size) { hidden_dim = new_size; }

// data type
using data_t = float;

void print(size_t n, data_t *a) {}

int *toDevice(const size_t n, int *x) {
    return x;
}

int *toHost(const size_t n, int *x) {
    return x;
}

data_t *toDevice(const size_t n, data_t *x) {
    return x;
}

data_t *toHost(const size_t n, data_t *x) {
    return x;
}

void fill_uniform(size_t n, data_t *a, const data_t L, const data_t R) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<data_t> dist(L, R);
    for (size_t i = 0; i < n; ++i) {
        a[i] = dist(gen);
    }
}

void fill_val(size_t n, data_t *a, const data_t val) {
    for(size_t i = 0; i < n; ++i) {
        a[i] = val;
    }
}

data_t sigmoid(data_t x) {
    return 1.0 / (1.0 + exp(-x));
}

data_t accuracy(const size_t batch_size, data_t *pred, int *label) {
    int correct = 0;
    for(int i = 0; i < batch_size; ++i) {
        int argmax = -1;
        for(int j = 0; j < output_dim; ++j) {
            if(argmax == -1 || pred[i * output_dim + j] > pred[i * output_dim + argmax])
                argmax = j;
        }
        if(argmax == label[i]) correct++;
    }
    return (data_t)correct / batch_size;
}


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
        ~model() {
            free(w_1);
            free(w_2);
            free(sum_1);
            free(sum_2);
            free(bias_1);
            free(bias_2);
            free(grad_w_1);
            free(grad_w_2);
            free(grad_loss);
            free(prev_grad);
            free(grad_bias_1);
            free(grad_bias_2);
            free(grad_sigmoid_1);
            free(grad_sigmoid_2);
        }
        model() {
            // first linear layer
            sum_1 = (data_t*)malloc(input_dim * sizeof(data_t));
            bias_1 = (data_t*)malloc(hidden_dim * sizeof(data_t));
            w_1 = (data_t*)malloc(input_dim * hidden_dim * sizeof(data_t));
            grad_sigmoid_1 = (data_t*)malloc(hidden_dim * sizeof(data_t));
            grad_w_1 = (data_t*)malloc(input_dim * hidden_dim * sizeof(data_t));
            grad_bias_1 = (data_t*)malloc(hidden_dim * sizeof(data_t));
            // second linear layer
            sum_2 = (data_t*)malloc(hidden_dim * sizeof(data_t));
            bias_2 = (data_t*)malloc(output_dim * sizeof(data_t));
            w_2 = (data_t*)malloc(hidden_dim * output_dim * sizeof(data_t));
            grad_sigmoid_2 = (data_t*)malloc(output_dim * sizeof(data_t));
            grad_w_2 = (data_t*)malloc(hidden_dim * output_dim * sizeof(data_t));
            grad_bias_2 = (data_t*)malloc(output_dim * sizeof(data_t));
            // loss 
            grad_loss = (data_t*)malloc(output_dim * sizeof(data_t));
            prev_grad = (data_t*)malloc(hidden_dim * sizeof(data_t));
        }
        data_t *forward(size_t batch_size, data_t *x) {
            batch_count += batch_size;
            // first layer forward
            data_t *temp_1 = (data_t*)malloc(batch_size * hidden_dim * sizeof(data_t));
            for(int i = 0; i < batch_size; ++i) {
                for(int j = 0; j < hidden_dim; ++j) {
                    temp_1[i * hidden_dim + j] = 0;
                    for(int k = 0; k < input_dim; ++k) {
                        temp_1[i * hidden_dim + j] += x[i * input_dim + k] * w_1[k * hidden_dim + j];
                    }
                }
            }
            // add bias
            for(int i = 0; i < batch_size; ++i) {
                for(int j = 0; j < hidden_dim; ++j) {
                    temp_1[i * hidden_dim + j] += bias_1[j];
                }
            }
            // sigmoid
            for(int i = 0; i < batch_size; ++i) {
                for(int  j = 0; j < hidden_dim; ++j) {
                    temp_1[i * hidden_dim + j] = sigmoid(temp_1[i * hidden_dim + j]);
                    grad_sigmoid_1[j] += temp_1[i * hidden_dim + j] * (1.0 - temp_1[i * hidden_dim + j]);
                }
            }
            // sum up for later calculation
            for(int i = 0; i < batch_size; ++i) {
                for(int j = 0; j < input_dim; ++j) {
                    sum_1[j] += x[i * input_dim + j];
                }
            }
            // second layer forward
            data_t *temp_2 = (data_t*)malloc(batch_size * output_dim *  sizeof(data_t));
            for(int i = 0; i < batch_size; ++i) {
                for(int j = 0; j < output_dim; ++j) {
                    temp_2[i * output_dim + j] = 0.0;
                    for(int k = 0; k < hidden_dim; ++k) {
                        temp_2[i * output_dim + j] += temp_1[i * hidden_dim + k] * w_2[k * output_dim + j];
                    }
                }
            }
            // add bias
            for(int i = 0; i < batch_size; ++i) {
                for(int j = 0; j < output_dim; ++j) {
                    temp_2[i * output_dim + j] += bias_2[j];
                }
            }
            // sigmoid
            for(int i = 0; i < batch_size; ++i) {
                for(int j = 0; j < output_dim; ++j) {
                    temp_2[i * output_dim + j] = sigmoid(temp_2[i * output_dim + j]);
                    grad_sigmoid_2[j] += temp_2[i * output_dim + j] * (1.0 - temp_2[i * output_dim + j]);
                }
            }
            // sum up for later calculation
            for(int i = 0; i < batch_size; ++i) {
                for(int j = 0; j < hidden_dim; ++j) {
                    sum_2[j] += temp_1[i * hidden_dim + j];
                }
            }
            free(temp_1);
            return temp_2;
        }
        void backward() {
            // grad of loss
            for(int i = 0; i < output_dim; ++i) {
                grad_loss[i] /= batch_count * output_dim / 2.0;
            }
            // grad of second sigmoid
            for(int i = 0; i < output_dim; ++i) {
                grad_sigmoid_2[i] /= batch_count;
                grad_sigmoid_2[i] *= grad_loss[i];
            }
            // grad of second bias
            for(int i = 0; i < output_dim; ++i) {
                grad_bias_2[i] = grad_sigmoid_2[i];
            }
            // grad of second w
            for(int i = 0; i < hidden_dim; ++i) {
                sum_2[i] /= batch_count;
            }
            for(int i = 0; i < hidden_dim; ++i) {
                prev_grad[i] = 0.0;
                for(int j = 0; j < output_dim; ++j) {
                    grad_w_2[i * output_dim + j] = sum_2[i] * grad_sigmoid_2[j];
                    prev_grad[i] += w_2[i * output_dim + j] * grad_sigmoid_2[j];
                }
            }
            // grad of first sigmoid
            for(int i = 0; i < hidden_dim; ++i) {
                grad_sigmoid_1[i] /= batch_count;
                grad_sigmoid_1[i] *= prev_grad[i];
            }
            // grad of first bias
            for(int i = 0; i < hidden_dim; ++i) {
                grad_bias_1[i] = grad_sigmoid_1[i];
            }
            // grad of first w
            for(int i = 0; i < input_dim; ++i) {
                sum_1[i] /= batch_count;
            }
            for(int i = 0; i < input_dim; ++i) {
                for(int j = 0; j < hidden_dim; ++j) {
                    grad_w_1[i * hidden_dim + j] = sum_1[i] * grad_sigmoid_1[j];
                }
            }
        }
        data_t loss(size_t batch_size, data_t *pred, data_t *real) {
            data_t temp, err = 0.0;
            for(int i = 0; i < batch_size; ++i) {
                for(int j = 0; j < output_dim; ++j) {
                    temp = pred[i * output_dim + j] - real[i * output_dim + j];
                    grad_loss[j] += temp;
                    err += temp * temp;
                }
            }
            return err / (data_t)(batch_size * output_dim);
        }
        void zero_grad() {
            batch_count = 0;
            fill_val(input_dim, sum_1, 0);
            fill_val(hidden_dim, sum_2, 0);
            fill_val(hidden_dim, grad_sigmoid_1, 0);
            fill_val(output_dim, grad_sigmoid_2, 0);
            fill_val(input_dim * hidden_dim, grad_w_1, 0);
            fill_val(hidden_dim * output_dim, grad_w_2, 0);
            fill_val(hidden_dim, grad_bias_1, 0);
            fill_val(output_dim, grad_bias_2, 0);
            fill_val(output_dim, grad_loss, 0);
            fill_val(hidden_dim, prev_grad, 0);
        }
        void update(data_t lr) {
            // update linear layer 1
            for(int i = 0; i < input_dim; ++i) {
                for(int j = 0; j < hidden_dim; ++j) {
                    w_1[i * hidden_dim + j] -= lr * grad_w_1[i * hidden_dim + j];
                }
            }
            for(int i = 0; i < hidden_dim; ++i) {
                bias_1[i] -= lr * grad_bias_1[i];
            }
            // update linear layer 2
            for(int i = 0; i < hidden_dim; ++i) {
                for(int j = 0; j < output_dim; ++j) {
                    w_2[i * output_dim + j] -= lr * grad_w_2[i * output_dim + j];
                }
            }
            for(int i = 0; i < output_dim; ++i) {
                bias_2[i] -= lr * grad_bias_2[i];
            }
        }
};

#endif