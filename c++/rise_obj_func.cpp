#include <iostream>
#include <cmath>
#include <bitset>
#include <time.h>
#include <fstream>
#include <string>
#include <inttypes.h>

using namespace std; 

const int n = 60; // n is immutable

struct rise_func {

	/* structure that contains the value
	of the objective function and its 
	gradient */

	double sn;
	double dsn[n];

};

rise_func rise_objective_func(uint64_t data[10000], int u){

	srand (time(NULL));

	 // number of variables

	int N = 10000; // number of data points

	uint64_t state = 0;
	uint64_t op = 0;

	int op_bits = 0;
	int eval_op = 0;

	double energy = 0;
	double sn = 0;
	double g[n];
	int eval_ops[n];

	rise_func rf;


	// get a random array of parameter values (for testing)
	for (int i = 0; i < n; i++) {
		g[i] = -1 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX/2);
		// cout << g[i] << endl;
	}

	for (int i = 0; i < N; i++) {

		state = data[i];

		energy = 0;

		for (int j = 0; j < n; j++) {

			// evaluate operators

			op = pow(2,u) + pow(2,j);
			op_bits = bitset<n>(op & state).count();
			op_bits %= 2;
			eval_op = 1 - 2 * op_bits;
			eval_ops[j] = eval_op;

			energy += g[j] * eval_op;
		}

	sn += exp(-energy)/N;

	for (int j = 0; j < n; j++) {

		rf.dsn[j] += (-eval_ops[j] * exp(-energy))/N;
	}

	}

	cout << sn << endl;

	rf.sn = sn;

	// for (int i = 0; i < n; i++) {
	// 	cout << rf.dsn[i] << endl;
	// }

	return rf;

}

int main() {

	string line, line_s;
	uint64_t nline = 0;
	uint64_t data[10000];
	int i = 0;

	ifstream datafile("test_data_n60_N10000.dat");

	while (getline (datafile, line)) {
		line_s = line.substr(0,n);
		nline = bitset<n>(line_s).to_ulong();
		data[i] = nline;
		i++;
	}

	rise_func rise = rise_objective_func(data, 0);

	return 0;
}