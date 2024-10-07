//
// Created by gebmer on 22.06.24.
//

#ifndef RHMPC_H
#define RHMPC_H

#include "pinocchio/autodiff/cppad.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"
#include <iostream>
#include <cppad/ipopt/solve.hpp>
#include "OCP.h"

namespace robocrane
{
    // constexpr int DOF_A = 7;
    // constexpr int DOF_U = 2;
    // constexpr int DOF = DOF_A + DOF_U;
    //
    // /* pinocchio types */
    // typedef double Scalar;
    // using Model = pinocchio::ModelTpl<Scalar>;
    // using Data = Model::Data;
    // using JointVector = Model::ConfigVectorType;
    // using TangentVector = Model::TangentVectorType;
    // using ControlVector = Eigen::Vector<double, DOF_A>;
    //
    // /* autodiff types */
    // typedef CppAD::AD<Scalar> ADScalar;
    // using ADModel = pinocchio::ModelTpl<ADScalar>;
    // using ADData = ADModel::Data;
    // using ADJointVector = ADModel::ConfigVectorType;
    // using ADTangentVector = ADModel::TangentVectorType;
    // using ADMotion = Model::Motion;
    // using VectorXAD = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;
    // using ADControlVector = Eigen::Vector<ADScalar, DOF_A>;
    typedef CPPAD_TESTVECTOR(Scalar) Dvector;

    class RHMPC
    {
        Model pin_model;
        Data pin_data;
        ADModel ad_model;
        ADData ad_data;
        int steps;
        Scalar dt;
        int tool_frame_id;

        // mpc horizons
        int n_split, n_new;
        const double goal_tolerance = 5e-3;

        // results, state constraints, dynamic constraints
        Eigen::MatrixXd q_result, qdot_result, u_result;
        JointVector q_lb, q_ub;
        JointVector qp_lb, qp_ub;
        ControlVector u_lb, u_ub;
        Eigen::MatrixXd g_dyn_ub, g_dyn_lb;

        std::vector<Scalar> weight_parameters;

        // stats
        int max_iterations, infeasible_iterations;

        int nx; // # of independent variables (domain dimension for f and g)
        int ng; // # of constraints (range dimension for g)
        Dvector xi; // initial value for the independent variables
        Dvector xl, xu; // lower and upper bounds for the independent variables
        Dvector gl, gu; // lower and upper bounds for the constraints
        std::unique_ptr<OCP> optimization_problem;

    public:
        RHMPC(const Model& pin_model, const Data& pin_data, const int steps, const double dt, JointVector q0,
              const int tool_frame_id);
        ~RHMPC();

        int iterate(JointVector &q0,
                    TangentVector &qdot0,
                    ControlVector &u0,
                    JointVector &q_waypoint,
                    JointVector &q_goal,
                    // std::vector<Scalar> &cost_weights_parameters,
                    bool use_actual_state = true);

        int ocp_generator(std::vector<Scalar> &cost_weights_parameters);

        void test_set_n_split(int n_split) { this->n_split = n_split; }
        void test_set_n_new(int n_new) { this->n_new = n_new; }

    private:
        void import_constraints_from_model();
        int intial_state_prediction(JointVector q0, TangentVector qdot0, ControlVector u0, ControlVector u1);
        int compute_equality_constraints();
        void reset_for_new_goal();
        int check_goal_distance(Eigen::MatrixXd q_sol, JointVector qd, const double tol, int start, int stop, int dof,
                                bool enable_print = false);
        void print_solver_statistics();

        void set_initial_state(JointVector q_init,
                               TangentVector qp_init,
                               ControlVector u_init);

        void set_constraints(
            JointVector q_init,
            TangentVector qp_init,
            ControlVector u_init,
            JointVector q_waypoint,
            JointVector q_goal,
            bool enable_terminal_constraints = false);
        void print_state_constraints(bool verbose);
    };
} // robocrane

#endif //RHMPC_H
