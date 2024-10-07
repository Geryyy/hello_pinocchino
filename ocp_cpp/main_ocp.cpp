
#include "pinocchio/autodiff/cppad.hpp"

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"

#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"

#include <iostream>
#include "Timer.h"

#include "rhtp/RHMPC.h"
#include "rhtp/OCP.h"


using namespace pinocchio;
using namespace Eigen;


typedef double Scalar;
typedef pinocchio::ModelTpl<Scalar> Model;
typedef Model::Data Data;
typedef Model::Motion Motion;

typedef Model::ConfigVectorType JointVector;
typedef Model::TangentVectorType TangentVector;


void print_frame_names(pinocchio::Model &model) {
    using namespace pinocchio;
    std::cout << "List of frame names in the model:" << std::endl;
    for (FrameIndex i = 0; i < model.frames.size(); ++i) {
        std::cout << "frame-id:" << i << ": " << model.frames[i].name << std::endl;
    }
}

void print_joint_positions(pinocchio::Model &model, pinocchio::Data &data) {
    using namespace pinocchio;
    // Print out the placement of each joint of the kinematic tree
    for (JointIndex joint_id = 0; joint_id < (JointIndex) model.njoints; ++joint_id)
        std::cout << std::setw(24) << std::left
                  << "id: " << joint_id << ", "
                  << model.names[joint_id] << ": "
                  << std::fixed << std::setprecision(2)
                  << data.oMi[joint_id].translation().transpose()
                  << std::endl;
}

void print_frame_positions(pinocchio::Model &model, pinocchio::Data &data) {
    using namespace pinocchio;
    // Print out the placement of each joint of the kinematic tree
    for (FrameIndex frame_id = 0; frame_id < model.frames.size(); ++frame_id)
        std::cout << std::setw(24) << std::left
                  << "id: " << frame_id << ", "
                  << model.frames[frame_id].name << ": "
                  << std::fixed << std::setprecision(2)
                  << data.oMf[frame_id].translation().transpose()
                  << std::endl;
}


int main(int argc, char **argv) {

    if (argc < 2) {
        std::cerr << "usage: pinopt <path_to_cpp_folder>" << std::endl;
        return -1;
    }
    std::string ws_path = std::string(argv[1]);
    const std::string urdf_filename = ws_path + std::string("model_dir/robocrane.urdf");

    Model model;
    pinocchio::urdf::buildModel(urdf_filename, model);
    Data data(model);

    std::cout << "Brief: Test case for RHMPC class." << std::endl;


    // sample random configuration, velocity, acceleration
    JointVector q(model.nq);
    q = pinocchio::randomConfiguration(model);

    constexpr int tool_frame_id = 54; // TODO: get this from the model by name
    print_frame_names(model);
    constexpr int steps = 2;
    robocrane::RHMPC mpc(model, data, steps, 0.1, q, tool_frame_id);

    std::cout << "Generate optimization problem" << std::endl;
    std::vector<Scalar> cost_weights_parameters = {1.,2.,3.,4.,5.,6.,7.};
    mpc.ocp_generator(cost_weights_parameters);

    std::cout << " run mpc iteration " << std::endl;
    JointVector q0 = q;
    TangentVector qdot0 = TangentVector::Zero(model.nv);
    robocrane::ControlVector u0 = robocrane::ControlVector::Zero(7);
    JointVector q_waypoint = q;
    JointVector q_goal = q;



    Timer timer;
    timer.tic();
    constexpr size_t N = 2;
    mpc.test_set_n_split(steps-1);
    mpc.test_set_n_new(steps-1);

    for(int i = 0; i < N; i++)
    {
        mpc.iterate(q0, qdot0, u0, q_waypoint, q_goal, true);
    }
    timer.toc();
    std::cout << "Elapsed time: " << timer.elapsed_time()/1e6/N << " ms" << std::endl;

    return 0;
}
