#include <iostream>
#include <string>
#include <Eigen/Dense>

using namespace std;

int main()
{
Eigen::MatrixXd a = Eigen::MatrixXd::Random(3,1);
a=(a+1)/2*60000;
Eigen::MatrixXi f = a.cast <int> ();   // Matrix of floats.}
std::cout<<f;
}
