#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "mnist/include/mnist/mnist_reader.hpp"
#include "mnist/include/mnist/mnist_utils.hpp"
// using Eigen::MatrixXd;
using namespace std;
using namespace Eigen;

auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
MatrixXd 				training_images(dataset.training_images.size(),dataset.training_images[0].size()),
								test_images(dataset.test_images.size(),dataset.training_images[0].size()),
								training_labels_unvect(dataset.training_labels.size(),1),
								test_labels_unvect (dataset.test_labels.size(),1),
								training_labels, test_labels;

MatrixXd vectorise_labels(MatrixXd labels){
	MatrixXd vectorised_labels = MatrixXd::Zero(labels.rows(), 10);
	for(int i = 0; i<vectorised_labels.rows(); i++)
		vectorised_labels(i,int(labels(i,0))) = 1;
	return vectorised_labels;
}

void load_dataset(){
	for(int i = 0; i<dataset.training_images.size(); i++)
		for(int j = 0; j<dataset.training_images[0].size(); j++)
			training_images(i,j) = double(dataset.training_images[i][j]);

	for(int i = 0; i<dataset.test_images.size(); i++)
		for(int j = 0; j<dataset.training_images[0].size(); j++)
			test_images(i,j) = double(dataset.test_images[i][j]);

	for(int i = 0; i<dataset.training_labels.size(); i++){
			training_labels_unvect(i,0) = double(dataset.training_labels[i]);
		}
		training_labels = vectorise_labels(training_labels_unvect);

	for(int i = 0; i<dataset.test_labels.size(); i++){
			test_labels_unvect(i,0) = double(dataset.test_labels[i]);
		}
		test_labels = vectorise_labels(test_labels_unvect);
}

class Layer{
public:
	Layer() = default;
	virtual MatrixXd forward_routine(const Ref<const MatrixXd>& inputs){}
	virtual MatrixXd backward_routine(const Ref<const MatrixXd>& gradients){}
	virtual void updateWeights(double, double){}
	virtual MatrixXd rloss(){}
};

class fc_layer : public Layer{

	int in_units, out_units;
	MatrixXd weights;
	Eigen::MatrixXd dw, db;
	MatrixXd inputs;
public:
	fc_layer(int input_units, int output_units){
		in_units = input_units;
		out_units = output_units;
	  weights = MatrixXd::Random(in_units,out_units)*0.01;
		}

virtual MatrixXd rloss(){
auto w = weights.transpose()*weights;
MatrixXd x(1,1);
x << w.array().sum();
return x;
}

virtual MatrixXd forward_routine(const Ref<const MatrixXd>& i){
		inputs = i;
		MatrixXd product = (inputs*weights);
	  return product;
	}

virtual MatrixXd backward_routine(const Ref<const MatrixXd>& ddot){
		MatrixXd dx;
		dw = (ddot.transpose()*inputs).transpose();
		dx = ddot*weights.transpose();
		return dx;
	}

virtual void updateWeights(double step_size, double reg){
	 dw = dw +(weights*reg);
	 weights = weights - step_size*dw;
}
};

/*
class conv_layer : public Layer{

	int in_units, out_units;
	int stride, padding;
	MatrixXd weights;
	MatrixXd bias;
	MatrixXd input;
public:
	conv_layer(int input_units, int output_units){
		in_units = input_units;
		out_units = output_units;
	  weights = MatrixXd::Random(in_units,out_units);
		}

virtual MatrixXd forward_routine(const Ref<const MatrixXd>& inputs){
		//product = product + bias;
		std::cout<<"called conv";
	}

virtual MatrixXd backward_routine(const Ref<const MatrixXd>& e){
 }
};
*/



class activation : public Layer
{
	int units;
	std::string option = "NULL";
	MatrixXd outputs;
	MatrixXd gradients;

public:
	virtual void updateWeights(double learning_rate, double reg){}
	virtual MatrixXd rloss(){}

	activation(int n, std::string x){
		units = n;
		option = x;
	}

	MatrixXd Sigmoid(MatrixXd x){
		MatrixXd firing_rate = x;
		firing_rate = 1/(1+(-1*firing_rate).array().exp());
		outputs = firing_rate;
		return firing_rate;
	}

	MatrixXd Tanh(MatrixXd x){
		outputs = x.array().tanh();
	  return outputs;
	}

	MatrixXd Relu(MatrixXd x){
	  for(int i = 0; i< x.rows(); i++){
	  for(int j = 0; j< x.cols(); j++){
	    if(x(i,j)>=0) continue;
	    else x(i,j) = 0;
	    }
	  }
		outputs = x;
	  return x;
	}

	MatrixXd Softmax(MatrixXd x){
		/*MatrixXd rt;
		for(int i = 0; i<x.rows(); i++)
		re.row(i) = x.row(i)/x.row(i).array().exp().sum();
		return rt.array().log(); */
		for(int i= 0; i< x.rows(); i++){
			double denom = x.row(i).array().exp().sum();
			for(int j = 0; j<x.cols(); j++){
				x(i,j) = exp(double(x(i,j)))/denom;
			}
		}
			outputs = x;
			return x;
	 }

	virtual MatrixXd forward_routine(const Ref<const MatrixXd>& inputs){
		if(option=="sigmoid") outputs = Sigmoid(inputs);
		else if(option=="tanh") outputs = Tanh(inputs);
	  else if(option=="relu") outputs = Relu(inputs);
	  else if(option=="softmax") outputs = Softmax(inputs);
	  else std::cout<<"Activation not supported."<<endl;
		return outputs;
	}

	MatrixXd Sigmoid_b(MatrixXd x){
		return outputs.array()*(MatrixXd::Ones(outputs.rows(),outputs.cols())-outputs).array();
	}

	MatrixXd Tanh_b(MatrixXd x){
	}

	MatrixXd Relu_b(MatrixXd x){
		MatrixXd derivative = x;
		for(int i = 0; i< outputs.rows(); i++){
			for(int j=0; j<outputs.cols(); j++){
				if(outputs(i,j) == 0) derivative(i,j) = 0;
			}
		}
		return derivative;
	}

	MatrixXd Softmax_b(MatrixXd labels){
		auto grad = outputs;
		for(int i=0; i<labels.cols(); i++){
			if(labels(0,i)==1) grad(0,i) -=1;
		}
		return grad;
	}

	virtual MatrixXd backward_routine(const Ref<const MatrixXd>& d){
		if(option=="sigmoid") return Sigmoid_b(d);
		else if(option=="tanh") return Tanh_b(d);
	  else if(option=="relu") return Relu_b(d);
	  else if(option=="softmax") return Softmax_b(d);
	  else std::cout<<"Activation not supported."<<endl;
	}
};

class Model
{
private:
  std::vector<Layer*> layers;
	double learning_rate, regularization, performance;
  MatrixXd input_data, outputs, in_gradient, gradients;
//protected:
public:
  Model() = default;
	int a;
  void add_layer(Layer* layer){
      layers.push_back(layer);
  }

	void setParams(double ss, double r){
		learning_rate = ss;
		regularization = r;
	}

	MatrixXd forward_prop(const Ref<const MatrixXd>& temp){
		outputs = temp;
	for(int i=0; i<layers.size(); i++)
		outputs = layers[i]->forward_routine(outputs);
		return outputs;
	}

	MatrixXd back_prop(const Ref<const MatrixXd>& temp){
	MatrixXd gradients = temp;
	for(int i = layers.size()-1; i>=0; i--){
		gradients = layers[i]->backward_routine(gradients);
	}
		return gradients;
	}

	virtual void updateWeights(double learning_rate, double reg){
		for(int i=0; i<layers.size(); i++)
			layers[i]->updateWeights(learning_rate, reg);
		}

	MatrixXd reg_loss(int batch){
	  MatrixXd rloss = MatrixXd::Zero(1,batch);
		for(int i=0; i<layers.size()-1;i++){
			rloss += layers[0]->rloss();
		}
		return rloss;
	}

	MatrixXd ce_loss(const Ref<const MatrixXd>& hx, const Ref<const MatrixXd>& y){
		MatrixXd data_loss;
		data_loss = -1*(y*(hx.transpose())).array().log();
		data_loss /= hx.cols();
		return data_loss;
	}

	double l2_loss(const Ref<const MatrixXd>& predictions, const Ref<const MatrixXd>& labels){
		return (labels-predictions).array().square().sum();
	}

	double max(MatrixXd vector){  //assumes a column vector is passed
		double m = vector(0,0);
		for(int i =0; i<vector.cols(); i++){
			if(vector(0,i)>m) m = vector(0,i);
		}
		return m;
	}

	double predict(MatrixXd outputs){
		for(int i = 0; i<outputs.rows(); i++){
			double maximum = max(outputs.row(i));
			for(int j=0; j<outputs.cols(); j++){
				if(outputs(i,j)<maximum) outputs(i,j) = 0;
				if(outputs(i,j)==maximum) outputs(i,j) = 1;
			}
		}

			for(int k= 0; k<outputs.cols(); k++){
				if(outputs(0,k)==1) return k;
			}
	}

	void train(int epochs, int batch){
		// select a (batch) number of pos elements from 0-60,000
		MatrixXd batch_trainimages(batch, training_images.row(0).cols()),
						 batch_trainlabels(batch, 1);


						for(int i=0; i<epochs; i++){
							std::cout<<"epoch:"<<epochs+1<<endl;
							for(int j=0; j<batch;j++){
								std::cout<<"image: "<<j+1<<endl;
								auto op = forward_prop(training_images.row(j));
								auto er = ce_loss(op, training_labels.row(j));
								auto rr = 0.5*regularization*reg_loss(batch);
								std::cout<<rr+er;
								auto p =  back_prop(training_labels.row(j)); // from the op vect, sub 1 where label is 1.
								updateWeights(learning_rate,regularization);
						}
					}
		}

	void test(){
		int score=0;
		for(int i=0; i<test_images.rows();i++){
			double p = predict(forward_prop(test_images.row(i)));
			if(double(test_labels_unvect(i,0))==p) ++score;
		}
		std::cout<<score<<endl;
		std::cout<<(double(score)/double(test_images.rows()));
	}
};

int main()
{
	Model neuralNet;
	load_dataset();

	float step_size = 1e-4, reg = 1e-3;

	fc_layer 		l0(784, 50);
	fc_layer 		l1(50, 10);
 	activation  l2(10,"softmax");

 	neuralNet.add_layer(&l0);
	neuralNet.add_layer(&l1);
        neuralNet.add_layer(&l2);

	neuralNet.setParams(step_size, reg);
	std::cout<<"Training..."<<endl;
	neuralNet.train(1,50);
	std::cout<<"Trained."<<endl;
	neuralNet.test();

	return 0;
}
