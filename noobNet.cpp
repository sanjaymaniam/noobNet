#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "mnist/include/mnist/mnist_reader.hpp"
#include "mnist/include/mnist/mnist_utils.hpp"

using Eigen::MatrixXd;
using namespace std;
using namespace Eigen;

auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
Eigen::MatrixXd training_images(dataset.training_images.size(),dataset.training_images[0].size()),
								test_images(dataset.test_images.size(),dataset.training_images[0].size()),
								training_labels(dataset.training_labels.size(),1),
								test_labels(dataset.test_labels.size(),1);

void load_dataset(){
	for(int i = 0; i<dataset.training_images.size(); i++)
		for(int j = 0; j<dataset.training_images[0].size(); j++)
			training_images(i,j) = double(dataset.training_images[i][j]);

	for(int i = 0; i<dataset.test_images.size(); i++)
		for(int j = 0; j<dataset.training_images[0].size(); j++)
			test_images(i,j) = double(dataset.test_images[i][j]);

	for(int i = 0; i<dataset.training_labels.size(); i++)
			training_labels(i,0) = double(dataset.training_labels[i]);

	for(int i = 0; i<dataset.test_labels.size(); i++)
			test_labels(i,0) = double(dataset.test_labels[i]);
}

class Layer{
public:
	Layer() = default;
	virtual Eigen::MatrixXd forward_routine(const Ref<const MatrixXd>& inputs){}
	virtual Eigen::MatrixXd backward_routine(const Ref<const MatrixXd>& gradients){}
};

class fc_layer : public Layer{

	int in_units, out_units;
	MatrixXd weights;
	Eigen::MatrixXd bias;
public:
	fc_layer(int input_units, int output_units){
		in_units = input_units;
		out_units = output_units;
	  weights = MatrixXd::Random(in_units,out_units);
		}

virtual Eigen::MatrixXd forward_routine(const Ref<const MatrixXd>& inputs){
		MatrixXd product = inputs*weights;
		// std::cout<<"called fc";
		// product = product + bias;
	  return product;
	}

virtual Eigen::MatrixXd backward_routine(const Ref<const MatrixXd>& inputs){
 }
};

class conv_layer : public Layer{

	int in_units, out_units;
	MatrixXd weights;
	Eigen::MatrixXd bias;
public:
	conv_layer(int input_units, int output_units){
		in_units = input_units;
		out_units = output_units;
	  weights = MatrixXd::Random(in_units,out_units);
		}

virtual Eigen::MatrixXd forward_routine(const Ref<const MatrixXd>& inputs){
		//product = product + bias;
		std::cout<<"called conv";
	}

virtual Eigen::MatrixXd backward_routine(const Ref<const MatrixXd>& inputs){

 }
};

class activation : public Layer
{
	int units;
	std::string option = "NULL";
	MatrixXd outputs;
public:
	activation(int n, std::string x){
		units = n;
		option = x;
	}

	Eigen::MatrixXd Sigmoid(Eigen::MatrixXd x){
		MatrixXd firing_rate = x;
		firing_rate = 1/(1+(-1*firing_rate).array().exp());
		return firing_rate;
	}

	Eigen::MatrixXd Tanh(Eigen::MatrixXd x){
	  return x.array().tanh();
	}

	Eigen::MatrixXd Relu(Eigen::MatrixXd x){
	  for(int i = 0; i< x.rows(); i++){
	  for(int j = 0; j< x.cols(); j++){
	    if(x(i,j)>=0) continue;
	    else x(i,j) = 0;
	    }
	  }
	  return x;
	}

	Eigen::MatrixXd Softmax(Eigen::MatrixXd x){
	  // std::cout<<"softmax?"<<endl;
	  Eigen::MatrixXd a = x.array().exp();
	  Eigen::MatrixXd b = x.array().exp().colwise().sum();
	  b = b.array().inverse();
	  x = a;//*b;
	  return x;
	 }

	virtual Eigen::MatrixXd forward_routine(const Ref<const MatrixXd>& inputs){
		if(option=="sigmoid") outputs = Sigmoid(inputs);
		else if(option=="tanh") outputs = Tanh(inputs);
	  else if(option=="relu") outputs = Relu(inputs);
	  else if(option=="softmax") outputs = Softmax(inputs);
	  else std::cout<<"Activation not supported."<<endl;
		return outputs;
	}

	Eigen::MatrixXd Sigmoid_b(Eigen::MatrixXd x){
	}

	Eigen::MatrixXd Tanh_b(Eigen::MatrixXd x){
	}

	Eigen::MatrixXd Relu_b(Eigen::MatrixXd x){
		Eigen::MatrixXd gradient = x;
		for(int i = 0; i< output.rows(); i++){
			for(int j=0; j<output.cols(); j++){
				if(output(i,j) == 0) gradient[i][j] = 0;
			}
		}
	}

	Eigen::MatrixXd Softmax_b(Eigen::MatrixXd x){

	}

	virtual Eigen::MatrixXd backward_routine(const Ref<const MatrixXd>& gradients){
		if(option=="sigmoid") return Sigmoid_b(inputs);
		else if(option=="tanh") return Tanh_b(inputs);
	  else if(option=="relu") return Relu_b(inputs);
	  else if(option=="softmax") return Softmax_b(inputs);
	  else std::cout<<"Activation not supported."<<endl;
	}
};

class Model
{
private:
  std::vector<Layer*> layers;
	double learning_rate, regularization, performance;
  MatrixXd input_data, output_data, in_gradient, out_gradient;
//protected:
public:
  Model() = default;
	int a;
  void add_layer(Layer* layer){
      layers.push_back(layer);
  }

	Eigen::MatrixXd forward_prop(const Ref<const MatrixXd>& temp)
	Eigen::MatrixXd outputs = temp;
	for(int i=0; i<layers.size(); i++
		outputs = layers[i]->forward_routine(outputs);

		return outputs;
	}

	Eigen::MatrixXd back_prop(const Ref<const MatrixXd>& temp){
	Eigen::MatrixXd gradients = temp;
	for(int i = layers.size(); i>=0; i--)
		gradients = layers[i]->backward_routine(gradients);

		return gradients;
	}

	virtual void updateWeights(){

	}

	double ce_loss(const Ref<const MatrixXd>& predictions, const Ref<const MatrixXd>& labels){
		Eigen::MatrixXd temp(labels.rows(),labels.cols());
		for(int i=0; i<labels.rows(); i++){
		for(int j=0; j<labels.cols(); j++){
			if(labels(i,j)==1) temp(i,j)= -1*log(predictions(i,j));
			else temp(i,j)= -1*log(1-predictions(i,j));
			}
		}
		double loss = temp.array().sum();
		return loss;
	}

	double l2_loss(const Ref<const MatrixXd>& predictions, const Ref<const MatrixXd>& labels){
		return (labels-predictions).array().square().sum();
	}
};

int main()
{
	Model neuralNet;
	load_dataset();

	fc_layer 		l0(784, 50);
	fc_layer 		l1(50, 10);
  // activation  l2(10,"softmax");
	activation  l2(10,"sigmoid");

  neuralNet.add_layer(&l0);
	neuralNet.add_layer(&l1);
  neuralNet.add_layer(&l2);
	// std::cout<<test_images.row(0);
	auto a = neuralNet.forward_prop(test_images.row(0));
  return 0;
}
