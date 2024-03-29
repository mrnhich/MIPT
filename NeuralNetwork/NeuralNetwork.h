#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <math.h>

using namespace std;

static double T = 10.0;

inline double randomRR(double minR, double maxR)
{
	return ((double) (rand() % 1000)) / 1000 * (maxR - minR) + minR;
}

inline double sqr(double x)
{
	return x * x;
}

// Sigmoid function
inline double ActivateFunction(double x)
{
	return 1.0 / (1.0 + exp(-x / T));
}

inline double DiffActFunc(double x)
{
	return x * (1.0 - x) / T;
}

class NeuralNetwork
{
public:
    NeuralNetwork(size_t number_of_layers, vector<size_t> layer_size);
	NeuralNetwork(string filePath);
	~NeuralNetwork();
	
	double calculateOutput(vector<double> input);
	void backPropagation(double output, double target);
	void learn(vector<double> input, double target, int i_train);
	void saveParametersToFile(string filePath);
private:
	size_t N;
	vector<vector<double> > y;
	vector<vector<double> > dy;
	vector<vector<vector<double> > > w;
	
    //ofstream flog;
};

#endif /* NEURALNETWORK_H */

