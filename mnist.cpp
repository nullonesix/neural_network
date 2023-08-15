// mnist is a dataset that consists of 28x28 pixel handwritten digit (0-9) images (grayscale)
// given an mnist image, the goal is to classify which of the 10 possible types it is
// one of the best ways to do this is with a convolutional neural network (CNN)

// intuitively the simplest neural network (NN) is one that is fully connected (FCNN)
// it's composed of a series of layers:
// data -> layer0 -> layer1 -> ... -> layerN -> output
// each layer is a matrix multiplication followed by an element-wise nonlinearity:
// nonlinearity(matrix * vector)
// the most common nonlinearity is the rectified linear (relu):
// relu := x if x > 0 else 0

// is layer0 is composed of 10 neurons and layer1 is composed of 10 neurons
// then the matrix between them is 10x10

// thus for mnist the we might have a fully connected network consisting of:
// 28*28 = 784 input layer neurons
// 10 middle layer neurons
// 10 output layer neurons:
// 784 -> 10 -> 10

// such a network will get about 90% - 95% accuracy
// which is much better than chance (10%)

// a CNN is different from fully connected in that it "convolves"
// learned patches/filters over the 2D input image (see https://youtu.be/f0t-OCG79-U)
// in this way it accounts for an images tendency to have some form of "spatial invariance"
// (an image of 7 translated is still a 7)
// thus a CNN can be seen as a pruning of a FCNN
// since the CNN has less edges than an FCNN, it learns more efficiently on images

// similarly a recurrent neural network (RNN) and a transformer (Bing/ChatGPT)
// neural architectures account for "temporal invariance"


#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64; // how many images are processed at the same time

// The batch size for testing.
const int64_t kTestBatchSize = 1000; // smaller batch sizes are better for training but at test time it doesn't matter

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10; // how many times the entire dataset is processed

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10; // how frequently to log

struct Net : torch::nn::Module {
  Net() // here we initialize our parameters
      : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)), // 2 dimensional convolution
      // passed over the 28x28 input image
      // with 1 input channel (grayscale), 10 filters/output channels, and a 5x5 filter size
      // see https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_conv2d_impl.html#classtorch_1_1nn_1_1_conv2d_impl
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        fc1(320, 50), // a fully connected layer with 320 input neurons and 50 output neurons
        fc2(50, 10) { // since there are 10 possible output classes (0-9)
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv2_drop", conv2_drop);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x) { // here we define propagation of information through the CNN
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2)); // see https://paperswithcode.com/method/max-pooling
    // since relu and max_pool2d are parameterless they are not declared at initialization
    x = torch::relu(
        torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
    x = x.view({-1, 320}); // converting to vector
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    // dropout is a "regularization technique" that randomly sets 0.5 of the neurons to be inactive
    // (with proper rescaling)
    // this helps prevent "overfitting" = memorizing the data
    // caveat: one of the things that separates NNs from other machine learning techinques
    // is that they can be made to not overfit regardless of how large the NN is 
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1); 
    // see https://en.wikipedia.org/wiki/Softmax_function
    // essentially a differentiable max function followed by a logarithm
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::Dropout2d conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

template <typename DataLoader>
void train(
    size_t epoch,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad(); // set gradients to zero
    auto output = model.forward(data); // propagate data through the model
    auto loss = torch::nll_loss(output, targets); // compute negative log likliehood loss
    // see https://notesbylex.com/negative-log-likelihood.html
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward(); // where the magic happens
    // loss = error
    // loss.backward() computes the derivative of the loss with respect to each parameter automatically!
    optimizer.step(); // step in the direction of less error in proportion to the learning rate (~= 0.01)

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     torch::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}

auto main() -> int {
  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  Net model;
  model.to(device); // put the model onto gpu if available (100x faster sometimes)

  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer( // stochastic gradient descent
  // gradient descent = running down a hill
  // "stochastic" because random batches are used
  // ie if batch_size = dataset_size then it would just be gradient descent
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }
}
