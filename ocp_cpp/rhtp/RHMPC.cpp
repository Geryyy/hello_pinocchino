//
// Created by gebmer on 22.06.24.
//

#include "RHMPC.h"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"

namespace robocrane {
    RHMPC::RHMPC(const Model &pin_model,
                 const Data &pin_data,
                 const int steps,
                 const double dt,
                 JointVector q0,
                 const int tool_frame_id) :
            pin_model(pin_model),
            pin_data(pin_data),
            steps(steps),
            dt(dt),
            tool_frame_id(tool_frame_id),
            n_split(steps),
            n_new(steps),
            weight_parameters(),
            max_iterations(0),
            infeasible_iterations(0) {
        // create auto diff structures
        ad_model = pin_model.cast<ADScalar>();
        ad_data = ADData(ad_model);

        // allocate and reset results
        q_result = Eigen::MatrixXd::Zero(steps, DOF);
        qdot_result = Eigen::MatrixXd::Zero(steps, DOF);
        u_result = Eigen::MatrixXd::Zero(steps, DOF_A);

        nx = steps * (2 * DOF + DOF_A);
        ng = 2 * (steps - 1) * DOF;

        xl.resize(nx);
        xu.resize(nx);
        gl.resize(ng);
        gu.resize(ng);
        xi.resize(nx);
    }

    RHMPC::~RHMPC() {
    }


    int RHMPC::ocp_generator(std::vector<Scalar> &cost_weights_parameters) {
        // state and control constraints
        import_constraints_from_model();

        // dynamic constraints
        int ng = 2 * (steps - 1) * DOF;
        auto g_dyn_ub = Eigen::VectorXd::Zero(ng); // TODO: move to OCP constructor, as it is always zero
        auto g_dyn_lb = Eigen::VectorXd::Zero(ng);

        std::cout << "Generating OCP..." << std::endl;
        std::cout << "q_ub: " << q_ub.transpose() << std::endl;
        std::cout << "q_lb: " << q_lb.transpose() << std::endl;
        std::cout << "qp_ub: " << qp_ub.transpose() << std::endl;
        std::cout << "qp_lb: " << qp_lb.transpose() << std::endl;
        std::cout << "u_ub: " << u_ub.transpose() << std::endl;
        std::cout << "u_lb: " << u_lb.transpose() << std::endl;
        std::cout << "g_dyn_ub.size: " << g_dyn_ub.size() << std::endl;
        std::cout << "g_dyn_lb.size: " << g_dyn_lb.size() << std::endl;

        weight_parameters = cost_weights_parameters;

        // print weight parameters
        for (int i = 0; i < weight_parameters.size(); i++) {
            std::cout << "RHMPC - weight_parameters[" << i << "] = " << weight_parameters[i] << std::endl;
        }

        auto ocp_problem = OCP(ad_model,
                               ad_data,
                               steps,
                               dt,
                               tool_frame_id,
                               weight_parameters,
                               n_split,
                               n_new
                               // g_dyn_ub, g_dyn_lb
                               );

        optimization_problem = std::make_unique<OCP>(ocp_problem);
        return 0;
    }

    int
    RHMPC::iterate(JointVector &q0,
                   TangentVector &qdot0,
                   ControlVector &u0,
                   JointVector &q_waypoint,
                   JointVector &q_goal,
                   // std::vector<Scalar> &cost_weights_parameters,
                   bool use_actual_state) {
        std::string options;
        // turn off any printing
        options += "Integer print_level  5\n";
        options += "String  sb           yes\n";
        // maximum number of iterations
        options += "Integer max_iter     300\n";
        // approximate accuracy in first order necessary conditions;
        // see Mathematical Programming, Volume 106, Number 1,
        // Pages 25-57, Equation (6)
        options += "Numeric tol          1e-3\n";
        // derivative testing
        options += "String  derivative_test            second-order\n";
        // maximum amount of random pertubation; e.g.,
        // when evaluation finite diff
        options += "Numeric point_perturbation_radius  0.\n";
        options += "String linear_solver ma57\n";


        q_waypoint = JointVector::Ones(DOF) * 0.5;
        q_goal = JointVector::Ones(DOF) * 0.8;

        // check if waypoint == goal
        if ((q_waypoint - q_goal).norm() < goal_tolerance) {
            std::cout << "Waypoint and goal are the same. n_split = 0." << std::endl;
            n_split = 0;
        }

        std::cout << "n_split: " << n_split << std::endl;
        std::cout << "n_new: " << n_new << std::endl;


        set_constraints(q0, qdot0, u0, q_waypoint, q_goal, true);
//        print_state_constraints(true);
        set_initial_state(q0, qdot0, u0);


        bool success = optimization_problem->solve(xi,
                                                   xu,
                                                   xl,
                                                   // cost_weights_parameters,
                                                   options);
        if (!success) {
            infeasible_iterations++;
            std::cout << "Infeasible iteration" << std::endl;
        } else {
            CppAD::ipopt::solve_result<OCP::Dvector> solution = optimization_problem->get_solution();
            std::cout << "Max iterations: " << max_iterations << std::endl;
            std::cout << "successful mpc iteration" << std::endl;

        }

        Dvector x_result = optimization_problem->get_solution().x;

        if (x_result.size() != steps * (2 * DOF + DOF_A)) {
            std::cerr << "unsuccessful mpc iteration" << std::endl;
            std::cerr << "Error: x_result.size() != steps * (2 * DOF + DOF_A)" << std::endl;
            return -1;
            // x_result.resize(steps * (2 * DOF + DOF_A));
        }
        // map x to q_result, qp_result, u_result
        for (int i = 0; i < steps; i++) {
            Eigen::VectorXd q_segment(DOF);
            std::copy(x_result.begin() + i * DOF, x_result.begin() + (i + 1) * DOF, q_segment.data());
            q_result.row(i) = q_segment;

            Eigen::VectorXd qdot_segment(DOF);
            std::copy(x_result.begin() + steps * DOF + i * DOF, x_result.begin() + steps * DOF + (i + 1) * DOF,
                      qdot_segment.data());
            qdot_result.row(i) = qdot_segment;

            Eigen::VectorXd u_segment(DOF_A);
            std::copy(x_result.begin() + 2 * steps * DOF + i * DOF_A,
                      x_result.begin() + 2 * steps * DOF + (i + 1) * DOF_A, u_segment.data());
            u_result.row(i) = u_segment;
        }

        // TODO: system dynamics
        // TODO: equality constraints



        // compute n_split: check if waypoint is within reach
        if (n_split == steps) {
            std::cout << "check waypoint distance" << std::endl;
            n_split = check_goal_distance(q_result, q_waypoint, goal_tolerance, 0, n_split, DOF_A);
        } else {
            n_split = std::max(n_split - 1, 0);
        }

        // compute n_new: check if goal is within reach
        if (n_new == steps) {
            std::cout << "check goal distance" << std::endl;
            n_new = check_goal_distance(q_result, q_goal, goal_tolerance, n_new, steps, DOF_A, true);
            n_new = std::max(n_new, 5);
        } else {
            if(n_split == 0)
            {
                n_new = std::max(n_new - 1, 2);
            }
        }





        // print solution
        std::cout << "Solution: " << optimization_problem->get_solution().x << std::endl;
        return success;
    }

    void RHMPC::import_constraints_from_model() {

        q_lb = pin_model.lowerPositionLimit;
        q_ub = pin_model.upperPositionLimit;
        qp_lb = -pin_model.velocityLimit;
        qp_ub = pin_model.velocityLimit;

        u_lb = -pin_model.effortLimit.block(0, 0, DOF_A, 1);
        u_ub = pin_model.effortLimit.block(0, 0, DOF_A, 1);
    }

    int RHMPC::check_goal_distance(Eigen::MatrixXd q_sol,
                                   JointVector qd,
                                   const double tol,
                                   int start,
                                   int stop,
                                   int dof,
                                   bool enable_print) {
        std::cout << "check_goal_distance" << std::endl;
        double d = INFINITY;
        bool reached[DOF] = {false};
        for (int i = start; i < stop; i++) {
            for (int j = 0; j < dof; j++) {
                d = q_sol(i, j) - qd(j);

                if (enable_print) {
                    std::cout << "d: " << d << std::endl;
                }

                if (std::abs(d) < tol) {
                    std::cout << "Goal reached" << std::endl;
                    reached[j] = true;
                }

                if (i > 0) {
                    if (std::signbit(q_sol(i, j) - qd(j)) !=
                        std::signbit(q_sol(i - 1, j) - qd(j))) {
                        std::cout << "Goal in between" << std::endl;
                        reached[j] = true;
                    }
                }
            }
        }
        // return true if all goals are reached
        bool goal_reached = true;
        for (int j = 0; j < dof; j++) {
            goal_reached &= reached[j];
        }
        return goal_reached;
    }


    void RHMPC::reset_for_new_goal() {
        n_split = steps;
        n_new = steps;
    }

    void RHMPC::set_initial_state(JointVector q_init,
                                  TangentVector qp_init,
                                  ControlVector u_init) {
        Eigen::Map<Eigen::MatrixXd> xi_map(xi.data(), 2 * DOF + DOF_A, steps);

        // TODO: initialize with q_init only at first step. in consecutive steps, initialize with previous ocp result.

        for (int m = 0; m < steps; m++) {
            xi_map.block(0, m, DOF, 1) = q_init;
            xi_map.block(DOF, m, DOF, 1) = qp_init;
            xi_map.block(2 * DOF, m, DOF_A, 1) = u_init;
        }
    }

    void RHMPC::set_constraints(
            JointVector q_init,
            TangentVector qp_init,
            ControlVector u_init,
            JointVector q_waypoint,
            JointVector q_goal,
            bool enable_terminal_constraints) {

        Eigen::Map<Eigen::MatrixXd> xl_map(xl.data(), 2 * DOF + DOF_A, steps);
        Eigen::Map<Eigen::MatrixXd> xu_map(xu.data(), 2 * DOF + DOF_A, steps);

        // --> set state constraints
        for (int m = 0; m < steps; m++) {
            // x = [q, qp, u].T

            // q
            xl_map.block(0, m, DOF, 1) = q_lb;
            xu_map.block(0, m, DOF, 1) = q_ub;

            // qp
            xl_map.block(DOF, m, DOF, 1) = qp_lb;
            xu_map.block(DOF, m, DOF, 1) = qp_ub;

            // u
            xl_map.block(2 * DOF, m, DOF_A, 1) = u_lb;
            xu_map.block(2 * DOF, m, DOF_A, 1) = u_ub;
        }

        // --> set initial state as constraint
        // q
        xl_map.block(0, 0, DOF, 1) = q_init;
        xu_map.block(0, 0, DOF, 1) = q_init;

        // qp
        xl_map.block(DOF, 0, DOF, 1) = qp_init;
        xu_map.block(DOF, 0, DOF, 1) = qp_init;

        // u
        xl_map.block(2 * DOF, 0, DOF_A, 1) = u_init;
        xu_map.block(2 * DOF, 0, DOF_A, 1) = u_init;

        // --> terminal constraints
        if (enable_terminal_constraints) {

            // set waypoint
            if (n_split > 1) {
                int waypoint_ind = n_split - 1;
                xl_map.block(0, waypoint_ind, DOF, 1) = q_waypoint;
                xu_map.block(0, waypoint_ind, DOF, 1) = q_waypoint;

                // TODO: set velocity at waypoint
            }

            // set goal
            if (n_new < steps) {
                // set goal at n_new
                int goal_ind = std::min(n_new + n_split - 1, steps - 1);
                xl_map.block(0, goal_ind, DOF, 1) = q_goal;
                xu_map.block(0, goal_ind, DOF, 1) = q_goal;

                // TODO: set velocities at goal

                // set control action at goal
                xl_map.block(2 * DOF, goal_ind, DOF_A, 1) = ControlVector::Zero(DOF_A);
                xu_map.block(2 * DOF, goal_ind, DOF_A, 1) = ControlVector::Zero(DOF_A);
            }
        }
    }


    void RHMPC::print_state_constraints(bool verbose = false) {
        Eigen::Map<Eigen::MatrixXd> xl_map(xl.data(), 2 * DOF + DOF_A, steps);
        Eigen::Map<Eigen::MatrixXd> xu_map(xu.data(), 2 * DOF + DOF_A, steps);

        std::cout << std::fixed;
        std::cout << std::setprecision(2);

        std::cout << "Step\t| q\t| qp\t| u \t UPPER LIMITS" << std::endl;

        for (int m = 0; m < steps; m++) {
            JointVector q = xu_map.block(0, m, DOF, 1);
            TangentVector qp = xu_map.block(DOF, m, DOF, 1);
            ControlVector u = xu_map.block(2 * DOF, m, DOF_A, 1);
            std::cout << m << "\t\t| ";
            std::cout << q.transpose() << "\t| ";
            std::cout << qp.transpose() << "\t| ";
            std::cout << u.transpose() << std::endl;
        }

        if (verbose) {
            std::cout << "Step\t| q\t| qp\t| u \t LOWER LIMITS" << std::endl;

            for (int m = 0; m < steps; m++) {
                JointVector q = xl_map.block(0, m, DOF, 1);
                TangentVector qp = xl_map.block(DOF, m, DOF, 1);
                ControlVector u = xl_map.block(2 * DOF, m, DOF_A, 1);
                std::cout << m << "\t\t| ";
                std::cout << q.transpose() << "\t| ";
                std::cout << qp.transpose() << "\t| ";
                std::cout << u.transpose() << std::endl;
            }
        }
    }


} // robocrane
