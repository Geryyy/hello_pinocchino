#include "pinocchio/codegen/cppadcg.hpp" // this file should be included first before all the others!
// #include <cppad/cppad.hpp>
// #include <cppad/cg/cppadcg.hpp>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/codegen/code-generator-algo.hpp"
//#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/rnea.hpp"

#include <iostream>
#include "Timer.h"

#define PIN_MODEL_DIR "/home/geraldebmer/repos/robocrane/cpp/"


int main(int argc, char ** argv) {
    using namespace pinocchio;
    using namespace Eigen;

    // typedef CppAD::eigen

    if(argc < 2)
    {
        std::cerr << "usage: pinopt <path_to_cpp_folder>" << std::endl;
        return -1;
    }
    std::string ws_path = std::string(argv[1]);
    const std::string urdf_filename = ws_path + std::string("model_dir/robocrane.urdf");

    Model model;
    pinocchio::urdf::buildModel(urdf_filename,model);

    std::cout << "Brief: Calculate derivatives with pinocchio and cppAD." << std::endl;

    CodeGenCRBA<double> crba_code_gen(model);

    // Generate the lib if it does not exist and load it afterwards.
    crba_code_gen.initLib();
    crba_code_gen.compileAndLoadLib(PINOCCHIO_CXX_COMPILER);

    // Use it with a random configuration samples in the bounds of the joint limits
    VectorXd q = randomConfiguration(model);
    crba_code_gen.evalFunction(q);

    // Retrieve the result
    MatrixXd & M = crba_code_gen.M;

    // And make it symmetric if needed
    M.template triangularView<Eigen::StrictlyLower>() = M.transpose().template triangularView<Eigen::StrictlyLower>();

    // You can check the result with the classic CRBA
    Data data_check(model);
    crba(model,data_check,q);

    data_check.M.triangularView<Eigen::StrictlyLower>() = data_check.M.transpose().triangularView<Eigen::StrictlyLower>();

    const MatrixXd &M_check = data_check.M;
    if(M_check.isApprox(M)) {
        std::cout << "Super! The two results are the same." << std::endl;
        return 0;
    }
    else
    {
        std::cout << "Not Super! The results do not match." << std::endl;
        return -1;
    }

    // return 0;
}
