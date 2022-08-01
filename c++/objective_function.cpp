#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <bitset>
#include <string>
#include <chrono>

using namespace std;
using namespace std::chrono;

#include <Eigen/Core>
#include <LBFGS.h>

using namespace Eigen;
using namespace LBFGSpp;

const int n = 10;
const string fname = "test_data_n10_N10000.dat";

vector<uint64_t> get_data(int *N);
vector<vector<double>> optimize(vector<uint64_t> data, int N);

/**********************************************************/
int main() {

	int N = 0;
	cout << "Reading data" << endl;
	auto data = get_data(&N);
	cout << "Optimizing" << endl;
	vector<vector<double>> jij = optimize(data, N);

	ofstream myfile;
	myfile.open("test_data_n10_N10000_jij_fit.dat");
	for (int i = 0; i < n; i++){
		for (int j = i + 1; j < n; j++){
			int op = pow(2,i) + pow(2,j);
			myfile << op << "\t" << jij[i][j] << "\n";
		}
	}
	myfile.close();

	return 0;
}

/**********************************************************/
vector<uint64_t> get_data(int *N) {

	// read data and get number of data points

	string line, subline;

	ifstream datafile(fname);

	while (getline (datafile, line)) {
		(*N)++;
	}

	const int size = (*N);

	vector<uint64_t> data(size);

	datafile.clear();
	datafile.seekg(0, datafile.beg);

	int i = 0;
	int nline; 

	while (getline (datafile, line)) {

		subline = line.substr(0,n);
		nline = bitset<n>(subline).to_ulong();
		data[i] = nline;
		i++;

	}

	return data;
}


/**********************************************************/
class rise_objective_func
{
private: 
	vector<uint64_t> data;
	int node; 
	int N; 
public:
	rise_objective_func(vector<uint64_t> data_, int node_, int N_) : data(data_), node(node_), N(N_) {}

	double operator()(const VectorXd& x, VectorXd& grad)

	{
		int op_bits, eval_op;
		uint64_t state, op;

		double sn = 0;
		double energy = 0;

		int eval_ops[n];
		
		// assign parameters and reset gradient
		double g[n];
		for (int i = 0; i < n; i++) {
			g[i] = x[i];
			grad[i] = 0;
		}
		g[node] = 0;

		for (int i = 0; i < N; i++) {

			state = data[i];

			energy = 0;

			for (int j = 0; j < n; j++) {

				// evaluate operators

				op = pow(2,node) + pow(2,j);
				op_bits = bitset<n>(op & state).count();
				op_bits %= 2;
				eval_op = 1 - 2 * op_bits;
				energy += g[j] * eval_op;

				eval_ops[j] = eval_op;

			}

			sn += exp(-energy)/N;

			for (int j = 0; j < n; j++) {

				grad[j] += (-eval_ops[j] * exp(-energy))/N;

			}

		}

		grad[node] = 0;

		return sn;
	}
};

/**********************************************************/
vector<vector<double>> optimize(vector<uint64_t> data, int N) {

	// initialize Jij
	vector<vector<double>> jij(n, vector<double>(n));

	// set up solver parameters
	LBFGSParam<double> param;
	param.epsilon = 1e-6;
	param.max_iterations = 500;

	// create solver object
	LBFGSSolver<double> solver(param);

	// optimize each node
	for (int i = 0; i < n; i++) {

		auto start = high_resolution_clock::now();
		cout << "Node: " << i;
		rise_objective_func rise(data, i, N);
		VectorXd g = VectorXd::Zero(n);

		double min_f; 
		int niter = solver.minimize(rise, g, min_f);

		for (int j = 0; j < n; j++){

			jij[i][j] = g[j];
		}

		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);

		cout << " in " << duration.count() << " msec" << endl;

	}

	// symmetrize jij 
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			jij[i][j] = (jij[i][j] + jij[j][i])/2;
			jij[j][i] = jij[i][j];

		}
	}

	return jij;
	

	
	
}