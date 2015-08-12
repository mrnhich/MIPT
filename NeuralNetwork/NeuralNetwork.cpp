#include "NeuralNetwork.h"

static double V = 0.1;

NeuralNetwork::NeuralNetwork(size_t number_of_layers, vector<size_t> layer_size)
{
	y.clear();
	dy.clear();
	w.clear();
	N = number_of_layers;
	
	// input layer
	vector<double> input_layer = vector<double>(layer_size.at(0) + 1, 0);
	input_layer.at(layer_size.at(0)) = 1.0;
	y.push_back(input_layer);
	dy.push_back(vector<double>());
	vector<vector<double> > w_0;
	for (size_t j = 0; j <= layer_size.at(0); ++j)
	{
		vector<double> w_0_j;
		for (size_t k = 0; k < layer_size.at(1); ++k)
		{
			w_0_j.push_back(randomRR(-0.5, 0.5));
		}
		w_0.push_back(w_0_j);
	}
	w.push_back(w_0);
	
	// hidden layers
	for (size_t i = 1; i < N - 1; ++i)
	{
		vector<double> hidden_layer = vector<double>(layer_size.at(i) + 1, 0.0);
		hidden_layer.at(layer_size.at(i)) = 1.0;
		y.push_back(hidden_layer);
		dy.push_back(vector<double>(layer_size.at(i) + 1, 0.0));
		
		vector<vector<double> > w_i;
		for (size_t j = 0; j <= layer_size.at(i); ++j)
		{
			vector<double> w_i_j;
			for (size_t k = 0; k < layer_size.at(i + 1); ++k)
			{
				w_i_j.push_back(randomRR(-0.5, 0.5));
			}
			w_i.push_back(w_i_j);
		}
		w.push_back(w_i);
	}

	// output layer
	y.push_back(vector<double>(layer_size.at(N - 1) + 1, 0.0));
	dy.push_back(vector<double>(layer_size.at(N - 1) + 1, 0.0));
	
    //flog.open("log.txt", ios::out);
}

NeuralNetwork::NeuralNetwork(string filePath)
{
	y.clear();
	dy.clear();
	w.clear();

	size_t _t;
	double _x;

	ifstream fi;
	fi.open(filePath.c_str(), ios::in);
	fi >> N;
	vector<size_t> layer_size;
	for (size_t i = 0; i < N; ++i)
	{
		fi >> _t;
		layer_size.push_back(_t);
	}
	// input layer
	vector<double> input_layer = vector<double>(layer_size.at(0) + 1, 0);
	input_layer.at(layer_size.at(0)) = 1.0;
	y.push_back(input_layer);
	dy.push_back(vector<double>());
	vector<vector<double> > input_layer_weight;
	for (size_t j = 0; j <= layer_size.at(0); ++j)
	{
		vector<double> wj;
		for (size_t k = 0; k < layer_size.at(1); ++k)
		{
			fi >> _x;
			wj.push_back(_x);
		}
		input_layer_weight.push_back(wj);
	}
	w.push_back(input_layer_weight);
	
	// hidden layers
	for (size_t i = 1; i < N - 1; ++i)
	{
		vector<double> hidden_layer = vector<double>(layer_size.at(i) + 1, 0.0);
		hidden_layer.at(layer_size.at(i)) = 1.0;
		y.push_back(hidden_layer);
		dy.push_back(vector<double>(layer_size.at(i) + 1, 0.0));
		
		vector<vector<double> > hidden_layer_weight;
		for (size_t j = 0; j <= layer_size.at(i); ++j)
		{
			vector<double> w_i_j;
			for (size_t k = 0; k < layer_size.at(i + 1); ++k)
			{
				fi >> _x;
				w_i_j.push_back(_x);
			}
			hidden_layer_weight.push_back(w_i_j);
		}
		w.push_back(hidden_layer_weight);
	}

	// output layer
	y.push_back(vector<double>(layer_size.at(N - 1) + 1, 0.0));
	dy.push_back(vector<double>(layer_size.at(N - 1) + 1, 0.0));

	fi.close();

    //flog.open("log.txt", ios::out);
}

NeuralNetwork::~NeuralNetwork()
{
    //flog.close();
}

double NeuralNetwork::calculateOutput(vector<double> input)
{
	for (size_t j = 0; j < input.size(); ++j)
	{
		y.at(0).at(j) = input.at(j);
	}
	for (size_t i = 1; i < N; ++i)
	{
		for (size_t k = 0; k < y.at(i).size() - 1; ++k)
		{
			double s = 0;
			for (size_t j = 0; j < y.at(i - 1).size(); ++j)
			{
				s += y.at(i - 1).at(j) * w.at(i - 1).at(j).at(k);
			}
			y.at(i).at(k) = i == N - 1 ? s : ActivateFunction(s);
		}
	}
	return y.at(N - 1).at(0);
}

void NeuralNetwork::backPropagation(double output, double target)
{
	// dy, output layer
	dy.at(N - 1).at(0) = target - output;

	// dy, hidden layers
    for (size_t i = N - 2; i > 0; --i)
	{
		for (size_t j = 0; j < y.at(i).size() - 1; ++j)
		{
            dy.at(i).at(j) = 0.0;
            for (size_t k = 0; k < y.at(i + 1).size() - 1; ++k)
            {
				dy.at(i).at(j) += dy.at(i + 1).at(k) * w.at(i).at(j).at(k);
            }
		}
	}

	// dw, output layer
	for (size_t j = 0; j < y.at(N - 2).size(); ++j)
	{
		w.at(N - 2).at(j).at(0) += V * dy.at(N - 1).at(0) * y.at(N - 2).at(j);
	}

	// dw, hidden layers
    for (size_t i = N - 2; i > 0; --i)
	{
		for (size_t j = 0; j < y.at(i).size() - 1; ++j)
		{
            double dif = DiffActFunc(y.at(i).at(j));
			
			for (size_t j_1 = 0; j_1 < y.at(i - 1).size(); ++j_1)
			{
                w.at(i - 1).at(j_1).at(j) += V * dy.at(i).at(j) * dif * y.at(i - 1).at(j_1);
			}
		}
	}
}

void NeuralNetwork::learn(vector<double> input, double target, int i_train)
{
	/*
	flog << "[ =" << i_train << "] Input = ";
	for (size_t i = 0; i < input.size(); ++i)
	{
		flog << input.at(i) << " ";
	}
	*/
	double output = calculateOutput(input);
	//flog << " >> [target] " << target << " >> [out 1] " << output;
	backPropagation(output, target);
	//flog << " >> [out 2] " << calculateOutput(input) << endl;
}

void NeuralNetwork::saveParametersToFile(string filePath)
{
	ofstream fo;
	fo.open(filePath.c_str(), ios::out);
	fo << N << endl;
	for (size_t i = 0; i < N; ++i)
	{
		fo << y.at(i).size() - 1 << " ";
	}
	fo << endl;
	for (size_t i = 0; i < N - 1; ++i)
	{
		for (size_t j = 0; j < y.at(i).size(); ++j)
		{
			for (size_t k = 0; k < y.at(i + 1).size() - 1; ++k)
			{
				fo << w.at(i).at(j).at(k) << " ";
			}
			fo << endl;
		}
	}
	fo.close();
}