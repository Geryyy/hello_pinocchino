#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio/algorithm/joint-configuration.hpp"
//#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/crba.hpp"
//#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/rnea.hpp"

#include <iostream>
#include "Timer.h"

#define PIN_MODEL_DIR "/home/geraldebmer/repos/robocrane/cpp/"


int main(int argc, char ** argv) {

    Timer timer1, timer2, timer3, timer4;
    int64_t duration1, duration2, duration3, duration4;

    const int N = 1e3;

    using namespace pinocchio;

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

    // Build a data related to model
    Data data(model);

    for(int i = 0; i<N; i++) {
        // Sample a random joint configuration as well as random joint velocity and torque
        Eigen::VectorXd q = randomConfiguration(model);
        Eigen::VectorXd v = randomConfiguration(model);//Eigen::VectorXd::Ones(model.nv);
        Eigen::VectorXd a = randomConfiguration(model);//Eigen::VectorXd::Ones(model.nv);
//        Eigen::VectorXd tau = Eigen::VectorXd::Zero(model.nv);

        // Allocate result container
        Eigen::MatrixXd djoint_acc_dq = Eigen::MatrixXd::Zero(model.nv, model.nv);
        Eigen::MatrixXd djoint_acc_dv = Eigen::MatrixXd::Zero(model.nv, model.nv);
        Eigen::MatrixXd djoint_acc_dtau = Eigen::MatrixXd::Zero(model.nv, model.nv);

        // Computes the forward dynamics (ABA) derivatives for all the joints of the robot
//    computeABADerivatives(model, data, q, v, tau, djoint_acc_dq, djoint_acc_dv, djoint_acc_dtau);

        timer1.tic();
        // Mass matrix
        pinocchio::crba(model, data, q);
        auto M_upper = data.M;
        Eigen::MatrixXd M_full = M_upper + M_upper.transpose();
        M_full.diagonal() -= M_upper.diagonal();
        duration1 += timer1.toc();

        // non-linear effects
        timer2.tic();
        auto C = pinocchio::computeCoriolisMatrix(model, data, q, v);
        auto g = pinocchio::computeGeneralizedGravity(model, data, q);
        auto nle_mat = C * v + g;
        duration2 += timer2.toc();

        timer3.tic();
        pinocchio::nonLinearEffects(model, data, q, v);
        duration3 += timer3.toc();

        auto diff_nle = nle_mat - data.nle;
//        std::cout << "Diff nle: " << diff_nle.norm() << std::endl;
//    std::cout << "nle_mat: " << nle_mat.transpose() << std::endl;
//    std::cout << "nle: " << data.nle.transpose() << std::endl;


        timer4.tic();
        auto Mu_inv = M_full.block<2, 2>(6, 6).inverse();
        auto Mua = M_full.block<2, 6>(6, 0);
        auto Cua = C.block<2, 6>(6, 0);
        auto Cu = C.block<2, 2>(6, 6);
        auto gu = g.segment(6, 2);

        auto a_a = a.segment(0, 6);
        auto v_a = v.segment(0, 6);
        auto v_u = v.segment(6, 2);


        // Compute the joint acceleration
        auto a_u = Mu_inv * (-Mua * a_a - Cua * v_a - Cu * v_u - gu);
        auto a_u_alt = Mu_inv * (-Mua * a_a - data.nle.segment(6, 2));
        duration4 += timer4.toc();

    }

    std::cout << "Time to compute CRBA: " << duration1/N << " ns" << std::endl;
    std::cout << "Time to compute non-linear effects: " << duration2/N << " ns" << std::endl;
    std::cout << "Time to compute non-linear effects (pinocchio): " << duration3/N << " ns" << std::endl;
    std::cout << "Time to compute joint acceleration: " << duration4/N << " ns" << std::endl;

//    std::cout << "M: " << M_full << std::endl;
//    std::cout << "C: " << C << std::endl;
//    std::cout << "g: " << g << std::endl;
//    std::cout << "a_u: " << a_u.transpose() << std::endl;
//    std::cout << "a_u_alt: " << a_u_alt.transpose() << std::endl;
//
//    std::cout << "diff: " << (data.nle.segment(6,2) - Cua*v_a - Cu*v_u - gu).transpose() << std::endl;


    // Get access to the joint acceleration
    std::cout << "Joint acceleration: " << data.ddq.transpose() << std::endl;


    return 0;
}
