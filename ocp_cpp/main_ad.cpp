
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

//#include "temp_file.h"


using namespace pinocchio;
using namespace Eigen;

using CppAD::AD;
using CppAD::NearEqual;

typedef double Scalar;
typedef AD<Scalar> ADScalar;

typedef pinocchio::ModelTpl<Scalar> Model;
typedef Model::Data Data;
typedef Model::Motion Motion;

typedef pinocchio::ModelTpl<ADScalar> ADModel;
typedef ADModel::Data ADData;

typedef Model::ConfigVectorType JointVector;
typedef Model::TangentVectorType TangentVector;

typedef ADModel::ConfigVectorType ADJointVector;
typedef ADModel::TangentVectorType ADTangentVector;

typedef ADModel::Motion ADMotion;

typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> VectorXAD;


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


void
computeTranslationalGeometricJacobian(ADModel &ad_model, ADData &ad_data, ADJointVector &ad_q, ADTangentVector &ad_v,
                                      ADTangentVector &ad_a, JointVector &q, FrameIndex frame_id) {
    CppAD::Independent(ad_q);
    pinocchio::forwardKinematics(ad_model, ad_data, ad_q, ad_v, ad_a);
    pinocchio::updateFramePlacements(ad_model, ad_data);

    SE3Tpl<ADScalar> oMf = ad_data.oMf[frame_id]; // Transformation matrix of the frame
    Vector3<ADScalar> position = oMf.translation(); // Position of the frame

    // Define dependent variable for CppAD
    VectorXAD y(3);
    for (int i = 0; i < 3; ++i) {
        y[i] = position[i];
    }

    std::cout << "ad_q: " << ad_q.transpose() << std::endl;

    // Create CppAD function
    CppAD::ADFun<Scalar> f(ad_q, y);

    CPPAD_TESTVECTOR(Scalar) q_test((size_t) ad_model.nq);
    for (int i = 0; i < ad_model.nq; ++i) {
        q_test[i] = q[i];
    }

    // Evaluate the function
    CPPAD_TESTVECTOR(Scalar) y_res = f.Forward(0, q_test);

    // Evaluate the Jacobian
    std::vector<Scalar> q_scalar(q.data(), q.data() + q.size());
    std::vector<Scalar> jac = f.Jacobian(q_scalar);
    // Print the Jacobian
    Eigen::Map<Eigen::Matrix<Scalar, 3, Eigen::Dynamic, Eigen::RowMajor>> jacobian_matrix(jac.data(), 3, ad_model.nq);
    std::cout << "CppAD Transl. Jacobian:\n" << jacobian_matrix << std::endl;
}


void computeRotationalGeometricJacobian(ADModel &ad_model, ADData &ad_data, ADJointVector &ad_q, ADTangentVector &ad_v,
                                        ADTangentVector &ad_a, JointVector &q, FrameIndex frame_id) {

    CppAD::Independent(ad_v);
    /* compute rotational geometric jacobian with CppAD */
    pinocchio::computeForwardKinematicsDerivatives(ad_model, ad_data, ad_q, ad_v, ad_a);
    std::cout << "v of joint 9: " << ad_data.v[9] << std::endl;

    ADMotion frame_velocity;
    frame_velocity = pinocchio::getFrameVelocity(ad_model, ad_data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED);

    VectorXAD y_rot(3);
//    Vector3<ADScalar> omega = ad_data.v[9].angular();
    Vector3<ADScalar> omega = frame_velocity.angular();
    for (int i = 0; i < 3; ++i) {
        y_rot[i] = omega[i];
    }

    // The input to the function to differentiate should be ad_v
    CppAD::ADFun<Scalar> frot(ad_v, y_rot);

    // Create a vector for the evaluation point, which should match ad_v's size
    std::vector<Scalar> v_scalar(ad_model.nv);
    for (int i = 0; i < ad_model.nv; ++i) {
        v_scalar[i] = Value(ad_v[i]); // Ensure v_scalar contains the actual numeric values of ad_v
    }

    // Compute the Jacobian with respect to ad_v (not ad_q)
    std::vector<Scalar> jac_rot = frot.Jacobian(v_scalar);

    // Print the Jacobian
    Eigen::Map<Eigen::Matrix<Scalar, 3, Eigen::Dynamic, Eigen::RowMajor>> jacobian_rot_matrix(jac_rot.data(), 3,
                                                                                              ad_model.nv);
    std::cout << "CppAD  Rot. Jacobian:\n" << jacobian_rot_matrix << std::endl;
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

    std::cout << "Brief: Calculate derivatives with pinocchio and cppAD." << std::endl;

    ADModel ad_model = model.cast<ADScalar>();
    ADData ad_data(ad_model);

    // sample random configuration, velocity, acceleration
    JointVector q(model.nq);
    q = pinocchio::randomConfiguration(model);
    TangentVector v(TangentVector::Random(model.nv));
    TangentVector a(TangentVector::Random(model.nv));

    ADJointVector ad_q = q.cast<ADScalar>();
    ADTangentVector ad_v = v.cast<ADScalar>();
    ADTangentVector ad_a = a.cast<ADScalar>();

    pinocchio::forwardKinematics(model, data, q, v, a);
    pinocchio::updateFramePlacements(model, data);

    std::cout << "print frame names" << std::endl;
    print_frame_names(model);
    std::cout << "joint positions: " << std::endl;
    print_joint_positions(model, data);
    std::cout << "frame positions: " << std::endl;
    print_frame_positions(model, data);


    // Extract the position of a specific frame (e.g., end-effector)
    const std::string frame_name = "base"; // Replace with the actual frame name
    FrameIndex frame_id = ad_model.getFrameId(frame_name);
    std::cout << "frame_id: " << frame_id << std::endl;

    // print position_scalar
    std::cout << "position_scalar: " << std::endl;
    std::cout << "frame transl: " << data.oMf[frame_id].translation().transpose() << std::endl;


    computeTranslationalGeometricJacobian(ad_model, ad_data, ad_q, ad_v, ad_a, q, frame_id);

    /* compute jacobian with pinocchio */
    pinocchio::computeJointJacobians(model, data, q);
    MatrixXd Jpin(6, model.nq);
    pinocchio::getFrameJacobian(model, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, Jpin);
    std::cout << "Transl. Jacobian with pinocchio:\n" << Jpin.block(0,0,3,model.nq) << std::endl;
//    std::cout << "Jacobian with pinocchio:\n" << Jpin << std::endl;

    /* compute rotational geometric jacobian with CppAD */
    computeRotationalGeometricJacobian(ad_model, ad_data, ad_q, ad_v, ad_a, q, frame_id);

    std::cout << "Rot. Jacobian with pinocchio:\n" << Jpin.block(3,0,3,model.nq) << std::endl;

    return 0;

}
