#include "NeuralNetwork.h"

using namespace std;

size_t input_layer_size = 2;
size_t number_of_layers = 3;
size_t hidden_layer_size = 10;

double minR = 0.0;
double maxR = 2 * M_PI;
int number_of_trains = 1000;
int number_of_samples = 10000;

const string dataSetFilePath = "target.txt";
const string testInputFilePath = "input.txt";
const string testOutputFilePath = "output.txt";
const string parametersFilePath = "parameters.txt";

NeuralNetwork * neural_network;

clock_t start, finish;

double targetFunction(vector<double> X)
{
	double f = 0.0;
	if (X.size() == 2)
	{
		f = sin(X.at(0)) * cos(X.at(0)) + sin(X.at(1)) * cos(X.at(1));
	}
	return f;
}

void generateDataSet(string filePath, bool calculate_output = true)
{
	ofstream fo;
	fo.open(filePath.c_str(), ios::out);
	fo << number_of_samples << " " << input_layer_size << endl;
	for (int i = 0; i < number_of_samples; ++i)
	{
		vector<double> X;
		for (size_t j = 0; j < input_layer_size; ++j)
		{
			double x = randomRR(minR, maxR);
			fo << x << " ";
			X.push_back(x);
		}
		if (calculate_output)
		{
			double y = targetFunction(X);
			fo << y;
		}
		fo << endl;
	}
	fo.close();
}

void createNeuralNetwork()
{
	vector<size_t> layer_size;
	layer_size.push_back(input_layer_size);
	for (size_t i = 1; i < number_of_layers - 1; ++i)
	{
		layer_size.push_back(hidden_layer_size);
	}
	layer_size.push_back(1);
    neural_network = new NeuralNetwork(number_of_layers, layer_size);
}

void trainFile(string filePath, int i_train)
{
	ifstream fi;
	fi.open(filePath.c_str(), ios::in);
	fi >> number_of_samples >> input_layer_size;
	for (int i = 0; i < number_of_samples; ++i)
	{
		vector<double> X;
		double x;
		double y;
		for (size_t j = 0; j < input_layer_size; ++j)
		{
			fi >> x;
			X.push_back(x);
		}
		fi >> y;
		neural_network->learn(X, y, i_train);
	}
	fi.close();
}

void trainNeuralNetwork()
{
	for (int i_train = 0; i_train < number_of_trains; ++i_train)
	{
		if (i_train % 10 == 9)
		{
			cout << "Loop : " << i_train + 1 << " / " << number_of_trains << endl;
		}
		trainFile(dataSetFilePath, i_train);
	}
}

void testNeuralNetwork(string inputFilePath, string outputFilePath)
{
	ifstream fi;
	ofstream fo;
	fi.open(inputFilePath.c_str(), ios::in);
	fo.open(outputFilePath.c_str(), ios::out);
	fi >> number_of_samples >> input_layer_size;
	fo << number_of_samples << " " << input_layer_size << endl;
	for (int i = 0; i < number_of_samples; ++i)
	{
		vector<double> X;
		double x;
		for (size_t j = 0; j < input_layer_size; ++j)
		{
			fi >> x;
			fo << x << " ";
			X.push_back(x);
		}
		double y = neural_network->calculateOutput(X);
		fo << y << endl;
	}
	fi.close();
	fo.close();
}

void testTraning()
{
	generateDataSet(dataSetFilePath);
	generateDataSet(testInputFilePath, false);
	
	createNeuralNetwork();
	trainNeuralNetwork();
	testNeuralNetwork(testInputFilePath, testOutputFilePath);
	neural_network->saveParametersToFile(parametersFilePath);
	delete neural_network;
}

void testParameters()
{
	neural_network = new NeuralNetwork(parametersFilePath);
	testNeuralNetwork(testInputFilePath, testOutputFilePath);
	delete neural_network;
}

void printTime(int c)
{
	long long clock_per_ms = (long long) CLOCKS_PER_SEC / 1000;
	long long clock_per_second = (long long) CLOCKS_PER_SEC;
	long long clock_per_minute = 60 * clock_per_second;
	long long clock_per_hour = 60 * clock_per_minute;
	long long z = 0;
	if 	(c > clock_per_hour)
	{
		z = 3;
	}
	else if (c > clock_per_minute)
	{
		z = 2;
	}
	else if (c > clock_per_second)
	{
		z = 1;
	}
	cout << "====================================================" << endl;
	cout << "Time = ";
	if (z >= 3)
	{
		cout << c / clock_per_hour << " hour(s) ";
		c %= clock_per_hour;
	}
	if (z >= 2)
	{
		cout << c / clock_per_minute << " minute(s) ";
		c %= clock_per_minute;
	}
	if (z >= 1)
	{
		cout << c / clock_per_second << " second(s) ";
		c %= clock_per_second;
	}
	cout << c / clock_per_ms << " ms(s)" << endl;
	cout << "====================================================" << endl;
}

int main(int argc, char *argv[])
{
	start = clock();
	testTraning();
	//testParameters();
	finish = clock();
	printTime(finish - start);
}

