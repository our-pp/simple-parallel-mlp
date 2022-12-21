#include "model.h"



data_t *toDevice(const size_t n, data_t *x) {
    data_t *ret;
    cudaMalloc(&ret, n * sizeof(data_t));
    cudaMemcpy(ret, x, n * sizeof(data_t), cudaMemcpyHostToDevice);
    free(x); 
    return ret;
}

int *toDevice(const size_t n, int *x) {
    int *ret;
    cudaMalloc(&ret, n * sizeof(int));
    cudaMemcpy(ret, x, n * sizeof(int), cudaMemcpyHostToDevice);
    free(x); 
    return ret;
}

data_t *toHost(const size_t n, data_t *x) {
    data_t *ret;
    ret = (data_t*)malloc(n * sizeof(data_t));
    cudaMemcpy(ret, x, n * sizeof(data_t), cudaMemcpyDeviceToHost);
    cudaFree(x);
    return ret;
} 

int *toHost(const size_t n, int *x) {
    int *ret;
    ret = (int*)malloc(n * sizeof(int));
    cudaMemcpy(ret, x, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(x);
    return ret;
} 

void print(size_t n, data_t *a) {
    data_t *temp = (data_t*)malloc(n * sizeof(data_t));
    cudaMemcpy(temp, a, n * sizeof(data_t), cudaMemcpyDeviceToHost);
    for(int i = 0; i < n; ++i) {
        printf("%f ", temp[i]);
    }
    printf("\n");
    free(temp);
}

void fill_uniform(size_t n, data_t *a, const data_t L, const data_t R) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<data_t> dist(L, R);
    for (size_t i = 0; i < n; ++i) {
        a[i] = dist(gen);
    }
}

__global__ void __fill__(size_t n, data_t *a, const data_t val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) a[i] = val;
}

void fill_val(size_t n, data_t *a, const data_t val) {
    int numBlocks = (n - 1) / 32 + 1;
    __fill__<<<numBlocks, 32>>>(n, a, val);
}

data_t sigmoid(data_t x) {
    return 1.0 / (1.0 + exp(-x));
}

__device__ data_t __sigmoid__(data_t x) {
    return 1.0 / (1.0 + exp(-x));
}

__global__ void __accuracy__(const size_t batch_size, data_t *pred, int *label, int *correct) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < batch_size) {
        int argmax = -1;
        for(int j = 0; j < output_dim; ++j) {
            if(argmax == -1 || pred[i * output_dim + j] > pred[i * output_dim + argmax])
                argmax = j;
        }
        if(argmax == label[i]) {
            atomicAdd(correct, 1);
        }
    }
} 

data_t accuracy(const size_t batch_size, data_t *pred, int *label) {
    int correct = 0;
    int *device_correct;
    cudaMalloc(&device_correct, sizeof(int));
    cudaMemset(device_correct, 0, sizeof(int));
    int numBlocks = (batch_size - 1) / 32 + 1;
    __accuracy__<<<numBlocks, 32>>>(batch_size, pred, label, device_correct);
    cudaMemcpy(&correct, device_correct, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_correct);
    return (data_t)correct / batch_size;
}

// some functions

// c = a * b
__global__ void __mul__(size_t n, size_t m, size_t p, data_t *a, data_t *b, data_t *c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < p) {
        c[i * p + j] = 0.0;
        for(int k = 0; k < m; ++k) {
            c[i * p + j] += a[i * m + k] * b[k * p + j];
        }
    }
}

// b[i, j] += a[j]
__global__ void __add__(size_t n, size_t m, data_t *a, data_t *b) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
        b[i * m + j] += a[j];
    }
}

// b[j] += a[i, j]
__global__ void __accumulate__(size_t n, size_t m, data_t *a, data_t *b) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if(j < m) {
        for(int i = 0; i < n; ++i) {
            b[j] += a[i * m + j];
        }
    }
}

// a = sigmoid(a) and accumulate its gradient
__global__ void __sigmoid__(size_t n, size_t m, data_t *a, data_t *grad) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
        a[i * m + j] = __sigmoid__(a[i * m + j]);
        atomicAdd(grad + j, a[i * m + j] * (1.0 - a[i * m + j]));
    }
}

__global__ void __div__(size_t n, data_t *a, data_t val) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n) {
        a[i] /= val;
    }
}

// c[i] = a[i] * b[i]
__global__ void __mul__(size_t n, data_t *a, data_t *b, data_t *c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n) {
        c[i] = a[i] * b[i];
    }
}

// b = a;
__global__ void __copy__(size_t n, data_t *a, data_t *b) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n) {
        b[i] = a[i];
    }
}

// c = dot(a, b)
__global__ void __dot__(size_t n, size_t m, data_t *a, data_t *b, data_t *c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < n && j < m) {
        c[i * m + j] = a[i] * b[j];
    }
}

// c[i] = sum(a[i, j] * b[j])
__global__ void __sum__(size_t n, size_t m, data_t *a, data_t *b, data_t *c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n) {
        c[i] = 0.0;
        for(int j = 0; j < m; ++j) {
            c[i] += a[i * m + j] * b[j];
        }
    }
}

// b -= ratio * a;
__global__ void __sub_eq__(size_t n, data_t *a, data_t *b, data_t ratio) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;      
    if(i < n) {
        b[i] -= a[i] * ratio;
    }
}

// model

model::~model()  {
    cudaFree(w_1);
    cudaFree(w_2);
    cudaFree(sum_1);
    cudaFree(sum_2);
    cudaFree(bias_1);
    cudaFree(bias_2);
    cudaFree(grad_w_1);
    cudaFree(grad_w_2);
    cudaFree(grad_loss);
    cudaFree(grad_bias_1);
    cudaFree(grad_bias_2);
    cudaFree(grad_sigmoid_1);
    cudaFree(grad_sigmoid_2);
}

model::model() {
    // init first layer
    bias_1 = (data_t*)malloc(hidden_dim * sizeof(data_t));
    fill_uniform(hidden_dim, bias_1, -sqrt(input_dim), sqrt(input_dim));
    bias_1 = toDevice(hidden_dim, bias_1);
    w_1 = (data_t*)malloc(input_dim * hidden_dim * sizeof(data_t));
    fill_uniform(input_dim * hidden_dim, w_1, -sqrt(input_dim), sqrt(input_dim));
    w_1 = toDevice(input_dim * hidden_dim, w_1);
    // init second layer
    bias_2 = (data_t*)malloc(output_dim * sizeof(data_t));
    fill_uniform(output_dim, bias_2, -sqrt(hidden_dim), sqrt(hidden_dim));
    bias_2 = toDevice(output_dim, bias_2);
    w_2 = (data_t*)malloc(hidden_dim * output_dim * sizeof(data_t));
    fill_uniform(hidden_dim * output_dim, w_2, -sqrt(hidden_dim), sqrt(hidden_dim));
    w_2 = toDevice(hidden_dim * output_dim, w_2);
    // first linear layer
    cudaMalloc(&sum_1, input_dim * sizeof(data_t));
    cudaMalloc(&grad_sigmoid_1, hidden_dim * sizeof(data_t));
    cudaMalloc(&grad_w_1, input_dim * hidden_dim * sizeof(data_t));
    cudaMalloc(&grad_bias_1, hidden_dim * sizeof(data_t));
    // second linear layer
    cudaMalloc(&sum_2, hidden_dim * sizeof(data_t));
    cudaMalloc(&grad_sigmoid_2, output_dim * sizeof(data_t));
    cudaMalloc(&grad_w_2, hidden_dim * output_dim * sizeof(data_t));
    cudaMalloc(&grad_bias_2, output_dim * sizeof(data_t));
    // loss 
    cudaMalloc(&grad_loss, output_dim * sizeof(data_t));
    cudaMalloc(&prev_grad, hidden_dim * sizeof(data_t));
}

data_t* model::forward(size_t batch_size, data_t *x) {
    batch_count += batch_size;
    // first layer forward
    data_t *temp_1;
    cudaMalloc(&temp_1, batch_size * hidden_dim * sizeof(data_t));
    {
        dim3 numBlocks((batch_size - 1) / 32 + 1, (hidden_dim - 1) / 32 + 1);
        dim3 numThreads(32, 32);
        __mul__<<<numBlocks, numThreads>>>(batch_size, input_dim, hidden_dim, x, w_1, temp_1);    
        // for(int i = 0; i < batch_size; ++i) {
        //     for(int j = 0; j < hidden_dim; ++j) {
        //         temp_1[i * hidden_dim + j] = 0;
        //         for(int k = 0; k < input_dim; ++k) {
        //             temp_1[i * hidden_dim + j] += x[i * input_dim + k] * w_1[k * hidden_dim + j];
        //         }
        //     }
        // }
    }
    // add bias
    {
        dim3 numBlocks((batch_size - 1) / 32 + 1, (hidden_dim - 1) / 32 + 1);
        dim3 numThreads(32, 32);
        __add__<<<numBlocks, numThreads>>>(batch_size, hidden_dim, bias_1, temp_1);
        // for(int i = 0; i < batch_size; ++i) {
        //     for(int j = 0; j < hidden_dim; ++j) {
        //         temp_1[i * hidden_dim + j] += bias_1[j];
        //     }
        // }
    }
    // sigmoid
    {
        dim3 numBlocks((batch_size - 1) / 32 + 1, (hidden_dim - 1) / 32 + 1);
        dim3 numThreads(32, 32);
        __sigmoid__<<<numBlocks, numThreads>>>(batch_size, hidden_dim, temp_1, grad_sigmoid_1);
        // for(int i = 0; i < batch_size; ++i) {
        //     for(int  j = 0; j < hidden_dim; ++j) {
        //         temp_1[i * hidden_dim + j] = sigmoid(temp_1[i * hidden_dim + j]);
        //         grad_sigmoid_1[j] += temp_1[i * hidden_dim + j] * (1.0 - temp_1[i * hidden_dim + j]);
        //     }
        // }
    }
    // sum up for later calculation
    {
        int numBlocks = (input_dim - 1) / 32 + 1; 
        __accumulate__<<<numBlocks, 32>>>(batch_size, input_dim, x, sum_1);
        // for(int i = 0; i < batch_size; ++i) {
        //     for(int j = 0; j < input_dim; ++j) {
        //         sum_1[j] += x[i * input_dim + j];
        //     }
        // }
    }
    // second layer forward
    data_t *temp_2;
    cudaMalloc(&temp_2, hidden_dim * output_dim *  sizeof(data_t));
    {   
        dim3 numBlocks((batch_size - 1) / 32 + 1, (output_dim - 1) / 32 + 1);
        dim3 numThreads(32, 32);
        __mul__<<<numBlocks, numThreads>>>(batch_size, hidden_dim, output_dim, temp_1, w_2, temp_2); 
        // for(int i = 0; i < batch_size; ++i) {
        //     for(int j = 0; j < output_dim; ++j) {
        //         temp_2[i * output_dim + j] = 0.0;
        //         for(int k = 0; k < hidden_dim; ++k) {
        //             temp_2[i * output_dim + j] += temp_1[i * hidden_dim + k] * w_2[k * output_dim + j];
        //         }
        //     }
        // }
    }
    // add bias
    {
        dim3 numBlocks((batch_size - 1) / 32 + 1, (output_dim - 1) / 32 + 1);
        dim3 numThreads(32, 32);
        __add__<<<numBlocks, numThreads>>>(batch_size, output_dim, bias_2, temp_2);
        // for(int i = 0; i < batch_size; ++i) {
        //     for(int j = 0; j < output_dim; ++j) {
        //         temp_2[i * output_dim + j] += bias_2[j];
        //     }
        // }
    }
    // sigmoid
    {
        dim3 numBlocks((batch_size - 1) / 32 + 1, (output_dim- 1) / 32 + 1);
        dim3 numThreads(32, 32);
        __sigmoid__<<<numBlocks, numThreads>>>(batch_size, output_dim, temp_2, grad_sigmoid_2);
        // for(int i = 0; i < batch_size; ++i) {
        //     for(int j = 0; j < output_dim; ++j) {
        //         temp_2[i * output_dim + j] = sigmoid(temp_2[i * output_dim + j]);
        //         grad_sigmoid_2[j] += temp_2[i * output_dim + j] * (1.0 - temp_2[i * output_dim + j]);
        //     }
        // }
    }
    // sum up for later calculation
    {
        int numBlocks = (hidden_dim - 1) / 32 + 1; 
        __accumulate__<<<numBlocks, 32>>>(batch_size, hidden_dim, temp_1, sum_2);
        // for(int i = 0; i < batch_size; ++i) {
        //     for(int j = 0; j < hidden_dim; ++j) {
        //         sum_2[j] += temp_1[i * hidden_dim + j];
        //     }
        // }
    }
    cudaFree(temp_1);
    return temp_2;
}

__global__ void __loss__(const size_t batch_size, data_t *pred, data_t *real, data_t *grad, data_t *err) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < batch_size) {
        for(int j = 0; j < output_dim; ++j) {
            data_t temp = pred[i * output_dim + j] - real[i * output_dim + j];
            atomicAdd(grad + j, temp);
            atomicAdd(err, temp * temp);
        }
    }
}

data_t model::loss(size_t batch_size, data_t *pred, data_t *real) {
    data_t err = 0.0;
    data_t *device_err;
    cudaMalloc(&device_err, sizeof(data_t));
    cudaMemset(device_err, 0, sizeof(data_t));
    int numBlocks = (batch_size - 1) / 32 + 1;
    __loss__<<<numBlocks, 32>>>(batch_size, pred, real, grad_loss, device_err);
    cudaMemcpy(&err, device_err, sizeof(data_t), cudaMemcpyDeviceToHost);
    cudaFree(device_err); 
    cudaFree(pred);
    return err / (data_t)(batch_size * output_dim);
    // data_t temp, err = 0.0;
    // for(int i = 0; i < batch_size; ++i) {
    //     for(int j = 0; j < output_dim; ++j) {
    //         temp = pred[i * output_dim + j] - real[i * output_dim + j];
    //         grad_loss[j] += temp;
    //         err += temp * temp;
    //     }
    // }
    // return err / (data_t)(batch_size * output_dim);
}

void model::zero_grad() {
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

void model::backward() {
    // grad of loss
    {
        int numBlocks = (output_dim - 1) / 32 + 1;
        __div__<<<numBlocks, 32>>>(output_dim, grad_loss, batch_count * output_dim / 2.0);
        // for(int i = 0; i < output_dim; ++i) {
        //     grad_loss[i] /= batch_count * output_dim / 2.0;
        // }
    }
    // grad of second sigmoid
    {
        int numBlocks = (output_dim - 1) / 32 + 1;
        __div__<<<numBlocks, 32>>>(output_dim, grad_sigmoid_2, batch_count);
        __mul__<<<numBlocks, 32>>>(output_dim, grad_sigmoid_2, grad_loss, grad_sigmoid_2);
        // for(int i = 0; i < output_dim; ++i) {
        //     grad_sigmoid_2[i] /= batch_count;
        //     grad_sigmoid_2[i] *= grad_loss[i];
        // }
        
    }
    // grad of second bias
    {
        int numBlocks = (output_dim - 1) / 32 + 1;
        __copy__<<<numBlocks, 32>>>(output_dim, grad_sigmoid_2, grad_bias_2);
        // for(int i = 0; i < output_dim; ++i) {
        //     grad_bias_2[i] = grad_sigmoid_2[i];
        // }
    }
    // grad of second w
    {
        int numBlocks = (hidden_dim - 1) / 32 + 1;
        __div__<<<numBlocks, 32>>>(hidden_dim, sum_2, batch_count);
        // for(int i = 0; i < hidden_dim; ++i) {
        //     sum_2[i] /= batch_count;
        // }
    }
    {
        dim3 numBlocks((hidden_dim - 1) / 32 + 1, (output_dim - 1) / 32 + 1);
        dim3 numThreads(32, 32);
        __dot__<<<numBlocks, numThreads>>>(hidden_dim, output_dim, sum_2, grad_sigmoid_2, grad_w_2);
        // for(int i = 0; i < hidden_dim; ++i) {
        //     for(int j = 0; j < output_dim; ++j) {
        //         grad_w_2[i * output_dim + j] = sum_2[i] * grad_sigmoid_2[j];
        //     }
        // }
    }
    {
        int numBlocks = (hidden_dim - 1) / 32 + 1;
        __sum__<<<numBlocks, 32>>>(hidden_dim, output_dim, w_2, grad_sigmoid_2, prev_grad);
        // for(int i = 0; i < hidden_dim; ++i) {
        //     prev_grad[i] = 0.0;
        //     for(int j = 0; j < output_dim; ++j) {
        //         prev_grad[i] += w_2[i * output_dim + j] * grad_sigmoid_2[j];
        //     }
        // }
    }
    
    // grad of first sigmoid
    {
        int numBlocks = (hidden_dim - 1) / 32 + 1;
        __div__<<<numBlocks, 32>>>(hidden_dim, grad_sigmoid_1, batch_count);
        __mul__<<<numBlocks, 32>>>(hidden_dim, grad_sigmoid_1, prev_grad, grad_sigmoid_1);
        // for(int i = 0; i < hidden_dim; ++i) {
        //     grad_sigmoid_1[i] /= batch_count;
        //     grad_sigmoid_1[i] *= prev_grad[i];
        // }
    }
    // grad of first bias
    {
        int numBlocks = (hidden_dim - 1) / 32 + 1;
        __copy__<<<numBlocks, 32>>>(hidden_dim, grad_sigmoid_1, grad_bias_1);
        // for(int i = 0; i < hidden_dim; ++i) {
        //     grad_bias_1[i] = grad_sigmoid_1[i];
        // }
    }
    // grad of first w
    {
        int numBlocks = (input_dim - 1) / 32 + 1;
        __div__<<<numBlocks, 32>>>(input_dim, sum_1, batch_count);
        // for(int i = 0; i < input_dim; ++i) {
        //     sum_1[i] /= batch_count;
        // }
    }
    {   
        dim3 numBlocks((input_dim - 1) / 32 + 1, (hidden_dim - 1) / 32 + 1);
        dim3 numThreads(32, 32);
        __dot__<<<numBlocks, numThreads>>>(input_dim, hidden_dim, sum_1, grad_sigmoid_1, grad_w_1);
        // for(int i = 0; i < input_dim; ++i) {
        //     for(int j = 0; j < hidden_dim; ++j) {
        //         grad_w_1[i * hidden_dim + j] = sum_1[i] * grad_sigmoid_1[j];
        //     }
        // }
    }
}

void model::update(data_t lr) {
    // update linear layer 1
    {
        int numBlocks = (input_dim * hidden_dim - 1) / 32 + 1;
        __sub_eq__<<<numBlocks, 32>>>(input_dim * hidden_dim, grad_w_1, w_1, lr);
        // for(int i = 0; i < input_dim; ++i) {
        //     for(int j = 0; j < hidden_dim; ++j) {
        //         w_1[i * hidden_dim + j] -= lr * grad_w_1[i * hidden_dim + j];
        //     }
        // }
    }
    {
        int numBlocks = (hidden_dim - 1) / 32 + 1;
        __sub_eq__<<<numBlocks, 32>>>(hidden_dim, grad_bias_1, bias_1, lr);
        // for(int i = 0; i < hidden_dim; ++i) { 
        //     bias_1[i] -= lr * grad_bias_1[i];
        // }
    }    
    // update linear layer 2
    {
        int numBlocks = (hidden_dim * output_dim - 1) / 32 + 1;
        __sub_eq__<<<numBlocks, 32>>>(hidden_dim * output_dim, grad_w_2, w_2, lr);
        // for(int i = 0; i < hidden_dim; ++i) {
        //     for(int j = 0; j < output_dim; ++j) {
        //         w_2[i * output_dim + j] -= lr * grad_w_2[i * output_dim + j];
        //     }
        // }
    }
    {
        int numBlocks = (output_dim - 1) / 32 + 1;
        __sub_eq__<<<numBlocks, 32>>>(output_dim, grad_bias_2, bias_2, lr);
        // for(int i = 0; i < output_dim; ++i) {
        //     bias_2[i] -= lr * grad_bias_2[i];
        // }
    }
}









