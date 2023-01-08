#include <getopt.h>
#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include "CycleTimer.h"

namespace nn = torch::nn;
using torch::Tensor;

const std::string MNIST_PATH = "/home/PP/PP-final/experiment/dataset/MNIST/raw";
const int DATASET_LENGTH = 60000;

struct Net : nn::Module {
    Net(size_t hidden_dim) : fc1{784, hidden_dim}, fc2{hidden_dim, 10} {}

    void forward(Tensor& x) {
        x = x.view({-1, 28 * 28});
        x = fc1(x);
        x = torch::sigmoid(x);
        x = torch::sigmoid(fc2(x));
    }

    void to(c10::Device device, bool non_blocking = false) {
        fc1->to(device, non_blocking);
        fc2->to(device, non_blocking);
    }

    nn::Linear fc1, fc2;
};

struct ModelParams {
    size_t batch_size = 8192;
    size_t hidden_dim = 300;
    size_t train_size = 60000;
    int epoch = 3;
    bool cpu_only = false;
};

ModelParams parse_args(int argc, char** argv);
torch::Device get_device(bool cpu_only);

int main(int argc, char** argv) {
    auto params = parse_args(argc, argv);

    auto device = get_device(params.cpu_only);

    auto dataset = torch::data::datasets::MNIST(MNIST_PATH)
                       .map(torch::data::transforms::Stack<>());

    auto data_loader =
        torch::data::make_data_loader(dataset, params.batch_size);

    nn::MSELoss loss_function;

    double start_time = CycleTimer::currentSeconds();

    auto model = std::make_shared<Net>(params.hidden_dim);
    model->to(device);

    torch::optim::SGDOptions options(0.01);
    options.momentum(0.0);
    auto optimizer = torch::optim::SGD(model->parameters(), options);

    for (int i = 1; i <= params.epoch; ++i) {
        model->train();
        double total_loss = 0;
        size_t trained_count = 0;

        for (auto& batch : *data_loader) {
            if (trained_count >= params.train_size) {
                break;
            }
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            optimizer.zero_grad();

            model->forward(data);

            auto target_onehot = torch::zeros(data.sizes())
                                     .to(device)
                                     .scatter(1, target.unsqueeze(-1), 1);

            auto loss = loss_function(data, target_onehot);
            loss.backward();
            total_loss += loss.data().item().toDouble();

            optimizer.step();
            trained_count += params.batch_size;
        }

        auto loss = total_loss / DATASET_LENGTH;
        printf("Epoch %d: Loss: %f, accuracy: %f\n", i, loss, 0.0);
    }

    double end_time = CycleTimer::currentSeconds();
    printf("time: %.4f sec\n", (end_time - start_time) / (double)params.epoch);

    return 0;
}

ModelParams parse_args(int argc, char** argv) {
    char opt;
    static struct option long_options[] = {
        {"batch-size", 1, nullptr, 'b'},  {"hidden-size", 1, nullptr, 'h'},
        {"epoch-count", 1, nullptr, 'e'}, {"train-size", 1, nullptr, 't'},
        {"cpu", 0, nullptr, 'c'},         {0, 0, 0, 0}};
    ModelParams params;

    while ((opt = getopt_long(argc, argv, "b:h:t:e:c", long_options,
                              nullptr)) != EOF) {
        switch (opt) {
            case 'b': {
                size_t new_size = std::stoul(optarg);
                params.batch_size = new_size;
                break;
            }
            case 'h': {
                size_t new_size = std::stoul(optarg);
                params.hidden_dim = new_size;
                break;
            }
            case 't': {
                size_t new_size = std::stoul(optarg);
                params.train_size = new_size;
                break;
            }
            case 'e': {
                int new_size = std::stoi(optarg);
                params.epoch = new_size;
                break;
            }
            case 'c': {
                params.cpu_only = true;
                break;
            }
        }
    }

    return params;
}

torch::Device get_device(bool cpu_only) {
    if (!torch::cuda::is_available() || cpu_only) {
        return torch::Device("cpu");
    }
    return torch::Device("cuda");
}
