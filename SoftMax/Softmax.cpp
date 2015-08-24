#include "Softmax.h"
#include <math.h>
#include <iostream>
using namespace std;
Softmax::Softmax(int _n_sample, int _n_input, int _n_output)
{
	n_sample = _n_sample;
	n_input = _n_input;
	n_output = _n_output;
	init();
}

void Softmax::init(){
	w = new double*[n_output];
	b = new double[n_output];
	for(int o=0;o<n_output;o++){
		w[o] = new double[n_input];
		for(int i = 0;i<n_input;i++){
			w[o][i] = 0;
		}
		b[o] = 0;
	}
}

void Softmax::softmax(double *_y){
	double max = 0.0;
	double sum = 0.0;
	//tim gia tri max
	for(int i = 0;i<n_output;i++){
		if(max < _y[i]) max = _y[i];
	}

	//tinh e^dy va sum(e^dy)
	for(int i=0;i<n_output;i++){
		_y[i] = exp(_y[i] - max);//e^dy
		sum += _y[i];
	}

	// tinh p(y=i|x)=e^dyi/sum(e^dyj) j tu 0 den k
	for(int i=0;i<n_output;i++){
		_y[i] = _y[i]/sum;
	}
}

void Softmax::train(double *_x,double *_y, double _learning_rate){
	double *y = new double[n_output];
	double *dy = new double[n_output];
	// tinh yi
	for(int i=0;i<n_output;i++){
		y[i] = 0.0;
		for(int j = 0;j<n_input;j++){
			y[i]+=w[i][j]*_x[j]; 
		}
		y[i]+=b[i];// y_i = sum_j(w_i*x_j+b_i)
	}
	softmax(y);
	for(int i=0;i<n_output;i++){
		dy[i] = _y[i] - y[i]; // y - y_softmax
		//update trong so
		for(int j=0;j<n_input;j++){
			w[i][j]+=_x[j] * dy[i] * _learning_rate/n_sample; // delta = x_j*(y_i - y_softmax)*learning_rate;
		}

		b[i] += _learning_rate*dy[i] / n_sample; // b_i gia tri thay doi cua lan hoc truoc
	}

	delete[] y;
	delete[] dy;
}

void Softmax::predict(double *_x,double *_y){
	// tinh yi
	for(int i=0;i<n_output;i++){
		_y[i] = 0.0;
		for(int j = 0;j<n_input;j++){
			_y[i]+=w[i][j]*_x[j]; 
		}
		_y[i]+=b[i];// y_i = sum_j(w_i*x_j+b_i)
	}
	softmax(_y);
}

void Softmax::predict_class(double *_x,int &_k){
	// tinh yi
	double *_y = new double[n_output];
	for(int i=0;i<n_output;i++){
		_y[i] = 0.0;
		for(int j = 0;j<n_input;j++){
			_y[i]+=w[i][j]*_x[j]; 
		}
		_y[i]+=b[i];// y_i = sum_j(w_i*x_j+b_i)
	}
	softmax(_y);
	double max = 0;
	int index_max = 0;
	for(int i = 0;i<n_output;i++){
		if(max < _y[i]){ 
			max = _y[i];
			index_max = i+1;
		}
	}
	_k = index_max;
}

void Softmax::test(){
	cout << "testing..." << endl;
  
	double learning_rate = 0.1;
	int n_epochs = 500;

	// training data
	double train_X[9][9] = {
		{1, 1, 1, 0, 0, 0, 0, 0, 0},// class 1
		{1, 0, 1, 0, 0, 0, 0, 0, 0},// class 1
		{1, 1, 0, 0, 0, 0, 0, 0, 0},// class 1
		{0, 0, 0, 1, 1, 1, 0, 0, 0},// class 2
		{0, 0, 0, 1, 0, 1, 0, 0, 0},// class 2
		{0, 0, 0, 1, 1, 0, 0, 0, 0},// class 2
		{0, 0, 0, 0, 0, 0, 1, 1, 1},// class 3
		{0, 0, 0, 0, 0, 0, 1, 0, 1},// class 3
		{0, 0, 0, 0, 0, 0, 1, 1, 0} // class 3
	};

	double train_Y[9][3] = {
		{1, 0, 0},
		{1, 0, 0},
		{1, 0, 0},
		{0, 1, 0},
		{0, 1, 0},
		{0, 1, 0},
		{0, 0, 1},
		{0, 0, 1},
		{0, 0, 1}
	};

	for(int i = 0;i<n_epochs;i++){
		for(int j=0;j<n_sample;j++){
			train(train_X[j],train_Y[j],learning_rate);
		}
	}
	// test data
	int _n_test = 3;
	double test_X[3][9] = {
		{0, 1, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 1, 1, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 1, 1}
	};

  double test_Y[3][3];
  // test

	for(int i=0; i<_n_test; i++) {
		predict(test_X[i], test_Y[i]);
		for(int j=0; j<n_output; j++) {
			cout << test_Y[i][j] << " ";
		}
		cout << endl;
	}
	cout << "testing class" << endl;
	int k;
	for(int i=0; i<_n_test; i++) {
		predict_class(test_X[i], k);
		cout << "{" << test_X[i][0];
		for(int j=1;j<n_input;j++){
			cout<<", " << test_X[i][j];
		}
		cout <<"}" << "-> class: "<< k;
		cout << endl;
	}
}

Softmax::~Softmax(void)
{
}