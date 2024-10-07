#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/mjcf.hpp>
#include "pinocchio/algorithm/model.hpp"
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Dense>

// Number of iterations for random sampling
const int N = 10;

int main() {
    // Load URDF model instead of MJCF (or adjust based on your XML file)
    std::string xml_file = "../../robot_model/robocrane.xml";  // Update with your URDF path
    // Create Pinocchio Model
    pinocchio::Model pin_model;
    pinocchio::mjcf::buildModelFromXML(xml_file, pin_model);
    pinocchio::Data pin_data(pin_model);

    std::cout << "Full DoF count: " << pin_model.nq << std::endl;

    // Print names of all joints
    for (size_t i = 0; i < pin_model.njoints; ++i) {
        std::cout << pin_model.names[i] << std::endl;
    }

    // Lock the joints from the 10th joint onwards (e.g. gripper joints)
    std::vector<std::string> joints_to_lock(pin_model.names.begin() + 10, pin_model.names.end());
    std::vector<pinocchio::JointIndex> joints_to_lock_ids;

    for (const auto& joint_name : joints_to_lock) {
        if (pin_model.existJointName(joint_name)) {
            joints_to_lock_ids.push_back(pin_model.getJointId(joint_name));
        } else {
            std::cerr << "Warning: joint " << joint_name << " does not belong to the model!" << std::endl;
        }
    }

    Eigen::VectorXd initial_joint_config = pinocchio::neutral(pin_model); //Eigen::VectorXd::Zero(pin_model.nq);
    pinocchio::Model pin_model_red = pinocchio::buildReducedModel(pin_model, joints_to_lock_ids, initial_joint_config);
    pinocchio::Data pin_data_red(pin_model_red);

    std::cout << "Reduced model nq: " << pin_model_red.nq << std::endl;

    // Random number generation for q, v, and a
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);  // Modify range as needed

    for (int i = 0; i < N; ++i) {
        // Sample random configurations
        Eigen::VectorXd q = Eigen::VectorXd::NullaryExpr(pin_model_red.nq, [&]() { return dis(gen); });
        Eigen::VectorXd v = Eigen::VectorXd::NullaryExpr(pin_model_red.nv, [&]() { return dis(gen); });
        Eigen::VectorXd a = Eigen::VectorXd::NullaryExpr(pin_model_red.nv, [&]() { return dis(gen); });

        // Perform kinematic and dynamic calculations using Pinocchio
        pinocchio::forwardKinematics(pin_model_red, pin_data_red, q, v, a);
        // pinocchio::computeAllTerms(pin_model_red, pin_data_red, q, v);

        // Forward kinematics for joint
        pinocchio::framesForwardKinematics(pin_model_red, pin_data_red, q);
        Eigen::Matrix4d H_joint = pin_data_red.oMi.back().toHomogeneousMatrix();

        // Frame placements
        pinocchio::updateFramePlacements(pin_model_red, pin_data_red);
        Eigen::Matrix4d H_body = pin_data_red.oMf.back().toHomogeneousMatrix();

        // Joint Jacobians
        pinocchio::computeJointJacobians(pin_model_red, pin_data_red, q);
        Eigen::MatrixXd J_joint = pin_data_red.J;

        // Inverse dynamics (RNEA)
        Eigen::VectorXd pin_tau = pinocchio::rnea(pin_model_red, pin_data_red, q, v, a);

        // Forward dynamics (ABA)
        Eigen::VectorXd acc_pin = pinocchio::aba(pin_model_red, pin_data_red, q, v, pin_tau);

        // Mass matrix (CRBA)
        Eigen::MatrixXd M_pin = pinocchio::crba(pin_model_red, pin_data_red, q);

        // Nonlinear effects (Coriolis and gravity)
        Eigen::VectorXd nle_pin = pinocchio::nonLinearEffects(pin_model_red, pin_data_red, q, v);

        // Coriolis matrix
        Eigen::MatrixXd C = pinocchio::computeCoriolisMatrix(pin_model_red, pin_data_red, q, v);

        // Gravity vector
        Eigen::VectorXd g = pinocchio::computeGeneralizedGravity(pin_model_red, pin_data_red, q);

        // Output some results to the console
        std::cout << "Iteration " << i + 1 << " results:" << std::endl;
        std::cout << "  tau: " << pin_tau.transpose() << std::endl;
        std::cout << "  acc: " << acc_pin.transpose() << std::endl;
    }

    return 0;
}
