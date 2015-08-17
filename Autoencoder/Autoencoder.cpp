#include "Autoencoder.h"

using namespace ___autoencoder___;

Autoencoder::Autoencoder(size_t number_of_layers, vector<size_t> layer_size)
{
    y.clear();
    dy.clear();
    w.clear();
    N = number_of_layers;

    for (size_t layer = 0; layer < N; ++layer)
    {
        vector<double> _y = vector<double>(layer_size.at(layer) + 1, 0.0);
        _y.at(layer_size.at(layer)) = 1.0;
        y.push_back(_y);
        dy.push_back(vector<double>(layer_size.at(layer), 0.0));

        // output layer >> break
        if (layer == N - 1)
        {
            break;
        }

        vector<vector<double> > _w;
        for (size_t j1 = 0; j1 < layer_size.at(layer) + 1; ++j1)
        {
            vector<double> _w_j;
            for (size_t j2 = 0; j2 < layer_size.at(layer + 1); ++j2)
            {
                _w_j.push_back(randomRR(-0.5, 0.5));
            }
            _w.push_back(_w_j);
        }
        w.push_back(_w);
    }
    sparsity_parameters = vector<double>(y.at(hidden_layer).size(), 0);
}

Autoencoder::Autoencoder(string file_path)
{
    y.clear();
    dy.clear();
    w.clear();

    size_t _t;
    double _x;

    ifstream fi;
    fi.open(file_path.c_str(), ios::in);
    fi >> N;
    vector<size_t> layer_size;
    for (size_t i = 0; i < N; ++i)
    {
        fi >> _t;
        layer_size.push_back(_t);
    }

    for (size_t layer = 0; layer < N; ++layer)
    {
        vector<double> _y = vector<double>(layer_size.at(layer) + 1, 0.0);
        _y.at(layer_size.at(layer)) = 1.0;
        y.push_back(_y);
        dy.push_back(vector<double>(layer_size.at(layer), 0.0));

        // output layer >> break
        if (layer == N - 1)
        {
            break;
        }

        vector<vector<double> > _w;
        for (size_t j1 = 0; j1 < layer_size.at(layer) + 1; ++j1)
        {
            vector<double> _w_j;
            for (size_t j2 = 0; j2 < layer_size.at(layer + 1); ++j2)
            {
                fi >> _x;
                _w_j.push_back(_x);
            }
            _w.push_back(_w_j);
        }
        w.push_back(_w);
    }
    fi.close();
    sparsity_parameters = vector<double>(y.at(hidden_layer).size(), 0);
}

Autoencoder::~Autoencoder()
{
}

void Autoencoder::prepare()
{
    bool debug = false;
    if (debug)
    {
        wcout << "prepare" << endl;
        wcout << "N = " << N << endl;
    }
    dw.clear();
    for (size_t layer = 0; layer < N - 1; ++layer)
    {
        if (debug)
        {
            wcout << "layer = " << layer << endl;
        }
        vector<vector<double> > _dw = vector<vector<double> >(y.at(layer).size() + 1,
                                      vector<double>(y.at(layer + 1).size(), 0));
        dw.push_back(_dw);
    }
    if (debug)
    {
        wcout << "prepare: 1" << endl;
    }
    fill(sparsity_parameters.begin(), sparsity_parameters.end(), 0);
    if (debug)
    {
        wcout << "prepare ok" << endl;
    }
}

void Autoencoder::calculateOutput(shared_ptr<vector<double> > input)
{
    bool debug = false;
    if (debug)
    {
        wcout << "calculateOutput..." << endl;
    }
    // Input layer
    for (size_t j = 0; j < input->size(); ++j)
    {
        y.at(0).at(j) = input->at(j);
    }

    // Hidden + output layers
    for (size_t layer = 1; layer < N; ++layer)
    {
        for (size_t j2 = 0; j2 < y.at(layer).size() - 1; ++j2)
        {
            double z = 0;
            for (size_t j1 = 0; j1 < y.at(layer - 1).size(); ++j1)
            {
                z += y.at(layer - 1).at(j1) * w.at(layer - 1).at(j1).at(j2);
            }
            y.at(layer).at(j2) = tanh(z);
        }
    }
}

void Autoencoder::backPropagation(shared_ptr<vector<double> > target)
{
    bool debug = false;
    if (debug)
    {
        wcout << "backPropagation..." << endl;
    }
    // dy, output layer
    for (size_t j = 0; j < y.at(N - 1).size() - 1; ++j)
    {
        if (debug)
        {
            wcout << "backPropagation: 1.1 j = " << j << endl;
        }
        dy.at(N - 1).at(j) = -(target->at(j) - y.at(N - 1).at(j))
                             * tanh_derivative(y.at(N - 1).at(j));
    }

    if (debug)
    {
        wcout << "backPropagation: 1" << endl;
    }

    // dy, hidden layers
    for (size_t layer = N - 2; layer > 0; --layer)
    {
        for (size_t j1 = 0; j1 < y.at(layer).size() - 1; ++j1)
        {
            dy.at(layer).at(j1) = 0.0;
            for (size_t j2 = 0; j2 < y.at(layer + 1).size() - 1; ++j2)
            {
                dy.at(layer).at(j1) += dy.at(layer + 1).at(j2)
                        * w.at(layer).at(j1).at(j2);
            }
            dy.at(layer).at(j1) *= tanh_derivative(y.at(layer).at(j1));
            if (layer == hidden_layer)
            {
                dy.at(layer).at(j1) += Beta * (-Rho / sparsity_parameters.at(j1)
                                     + (1 - Rho) / (1 - sparsity_parameters.at(j1)));
            }
        }
    }

    if (debug)
    {
        wcout << "backPropagation: 2" << endl;
    }

    // update dw
    for (size_t layer = N - 2; layer > 0; --layer)
    {
        for (size_t j1 = 0; j1 < y.at(layer - 1).size(); ++j1)
        {
            for (size_t j2 = 0; j2 < y.at(layer).size() - 1; ++j2)
            {
                dw.at(layer - 1).at(j1).at(j2) +=
                        dy.at(layer).at(j2) * y.at(layer - 1).at(j1);
            }
        }
    }

    if (debug)
    {
        wcout << "backPropagation: 3" << endl;
    }
}

void Autoencoder::updateWeight(int m)
{
    bool debug = false;
    if (debug)
    {
        wcout << "updateWeight..." << endl;
    }
    for (size_t layer = 1; layer < N; ++layer)
    {
        if (debug)
        {
            wcout << "layer = " << layer << endl;
        }
        for (size_t j2 = 0; j2 < y.at(layer).size() - 1; ++j2)
        {
            // Neurons
            for (size_t j1 = 0; j1 < y.at(layer - 1).size() - 1; ++j1)
            {
                if (debug)
                {
                    wcout << "j1 = " << j1 << "; j2 = " << j2 << endl;
                }
                w.at(layer - 1).at(j1).at(j2) -= Alpha * (dw.at(layer - 1).at(j1).at(j2) / m + Lambda * w.at(layer - 1).at(j1).at(j2));
            }
            // Bias
            if (debug)
            {
                wcout << "updateWeight:1" << endl;
            }
            w.at(layer - 1).at(y.at(layer - 1).size() - 1).at(j2) -= Alpha
                    * (dw.at(layer - 1).at(y.at(layer - 1).size() - 1).at(j2) / m);
            if (debug)
            {
                wcout << "updateWeight:2" << endl;
            }
        }
    }
    if (debug)
    {
        wcout << "updateWeight ok." << endl;
    }
}

void Autoencoder::trainFile(string file_path, bool calculate_sparsity, int iteration)
{
    bool debug = true;
    int m; // number of samples
    shared_ptr<vector<double> > input
            = make_shared<vector<double> >(y.at(0).size(), 0);

    ifstream fi;
    fi.open(file_path.c_str(), ios::in);
    fi >> m;
//    m = 2000;
    double x;
    for (int i = 0; i < m; ++i)
    {
        if (debug && i % 10000 == 0)
        {
            wcout << iteration << "." << i << endl;
        }
        for (size_t j = 0; j < y.at(0).size(); ++j)
        {
            fi >> x;
            input->at(j) = x;
        }
        calculateOutput(input);
        if (calculate_sparsity)
        {
            for (size_t j = 0; j < y.at(hidden_layer).size() - 1; ++j)
            {
                sparsity_parameters.at(j) += y.at(hidden_layer).at(j);
            }
        }
        else
        {
            backPropagation(input);
        }
    }
    if (calculate_sparsity)
    {
        for (size_t j = 0; j < y.at(hidden_layer).size() - 1; ++j)
        {
            sparsity_parameters.at(j) /= m;
        }
    }
    else
    {
        updateWeight(m);
    }
    fi.close();
}

void Autoencoder::train(string file_path, int iterations)
{
    bool debug = true;
    for (int i = 0; i < iterations; ++i)
    {
        if (debug)
        {
            wcout << "Iteration: " << i + 1 << endl;
        }
        prepare();
        trainFile(file_path, true, i + 1); // calculate sparsity parameters
        trainFile(file_path, false, i + 1); // update weight

        // Server
        string tmp_path = "Autoencoder_parameters_tmp//autoencoder_params_" + std::to_string(i) + ".txt";
        this->saveParametersToFile(tmp_path);
    }
}

void Autoencoder::saveParametersToFile(string file_path)
{
    ofstream fo;
    fo.open(file_path.c_str(), ios::out);
    fo << N << endl;
    for (size_t layer = 0; layer < N; ++layer)
    {
        fo << y.at(layer).size() - 1 << " ";
    }
    fo << endl;
    for (size_t layer = 0; layer < N - 1; ++layer)
    {
        for (size_t j1 = 0; j1 < y.at(layer).size(); ++j1)
        {
            for (size_t j2 = 0; j2 < y.at(layer + 1).size() - 1; ++j2)
            {
                fo << w.at(layer).at(j1).at(j2) << " ";
            }
            fo << endl;
        }
    }
    fo.close();
}
