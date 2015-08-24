#pragma once
class Softmax
{
public:
	Softmax(int _number_of_sample,int _number_of_input, int _number_of_output);
	~Softmax(void);
public:
	int n_sample;
	int n_input;
	int n_output;
	double **w;// ma tran (number_input x number_output)
	double *b; // gia tri sai so number_output chieu
	void init();// khoi tao gia tri ban dau
	void train(double *input, double * output, double learning_rate);
	void softmax(double *output);
	void predict(double *input, double *result);
	void predict_class(double *input, int &k);
	void test();
};

