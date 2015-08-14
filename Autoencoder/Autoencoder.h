#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <memory>
#include <math.h>

using namespace std;

namespace ___autoencoder___ {

static double Alpha = 0.1;
static double Rho = 0.2;
static double Beta = 0.3;
static double Lambda = 0.001;
static size_t hidden_layer = 1;

inline double randomRR(double minR, double maxR)
{
    return ((double) (rand() % 1000)) / 1000 * (maxR - minR) + minR;
}

inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

inline double sigmoid_derivative(double y)
{
    return y * (1.0 - y);
}

inline double tanh(double x)
{
    return (1.0 - exp(- 2 * x)) / (1.0 + exp(- 2 * x));
}

inline double tanh_derivative(double y)
{
    return 1.0 - y * y;
}

class Autoencoder
{
public:
    Autoencoder(size_t number_of_layers, vector<size_t> layer_size);
    Autoencoder(string file_path);
   ~Autoencoder();
    void prepare();
    void calculateOutput(shared_ptr<vector<double> > input);
    void backPropagation(shared_ptr<vector<double> > target);
    void updateWeight(int m);
    void trainFile(string file_path, bool calculate_sparsity);
    void train(string file_path, int iterations);
    void saveParametersToFile(string file_path);

private:
    size_t N; // number of layers
    vector<vector<double> > y;
    vector<vector<double> > dy;
    vector<vector<vector<double> > > w;
    vector<vector<vector<double> > > dw;
    vector<double> sparsity_parameters;
};

} // namespace Autoencoder
#endif /* AUTOENCODER_H */

