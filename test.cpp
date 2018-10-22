#include <iostream>
#include <string>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class a{
  MatrixXd alpha = MatrixXd::Random(1,4);
public:
  a(){
    std::cout<<alpha;
  }
};

int main()
{
  // Eigen::MatrixXd a(2,3), b(1,3);
  // std::cout<<MatrixXd::Random(1,4);
  a a1;
}
