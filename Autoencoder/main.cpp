#include "Autoencoder.h"

using namespace std;

using namespace ___autoencoder___;

Autoencoder * autoencoder;

clock_t start, finish;

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
    wcout << "====================================================" << endl;
    wcout << "Time = ";
	if (z >= 3)
	{
        wcout << c / clock_per_hour << " hour(s) ";
		c %= clock_per_hour;
	}
	if (z >= 2)
	{
        wcout << c / clock_per_minute << " minute(s) ";
		c %= clock_per_minute;
	}
	if (z >= 1)
	{
        wcout << c / clock_per_second << " second(s) ";
		c %= clock_per_second;
	}
    wcout << c / clock_per_ms << " ms(s)" << endl;
    wcout << "====================================================" << endl;
}

void test()
{
//    size_t vector_space = 50;
//    vector<size_t> layer_size;
//    layer_size.push_back(2 * vector_space);
//    layer_size.push_back(vector_space);
//    layer_size.push_back(2 * vector_space);
//    autoencoder = new Autoencoder(3, layer_size);

    // Server
    autoencoder = new Autoencoder("Autoencoder_parameters_tmp//autoencoder_params_0.txt");
    autoencoder->train("autoencoder_traning_set.txt", 1000);
    autoencoder->saveParametersToFile("autoencoder_parameters.txt");
}

int main(int argc, char *argv[])
{
	start = clock();
    test();
	finish = clock();
	printTime(finish - start);
}

