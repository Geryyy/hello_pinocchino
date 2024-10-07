//
// Created by gebmer on 22.06.24.
//

#include "OCP.h"

namespace robocrane {
    OCP::OCP(ADModel &cpin_model,
             ADData &cpin_data,
             const int steps,
             const Scalar step_size,
             const int tool_frame_id,
             const std::vector<Scalar> &weight_parameters,
             int &n_split, int &n_new
             // MatrixXd g_dyn_ub, MatrixXd g_dyn_lb
             ) :
            fg_eval(cpin_model,
                    cpin_data,
                    steps,
                    step_size,
                    tool_frame_id,
                    weight_parameters,
                    n_split,
                    n_new
            ),
            cmodel(cpin_model),
            cdata(cpin_data),
            // g_dyn_ub(g_dyn_ub),
            // g_dyn_lb(g_dyn_lb),
            steps(steps) {
        // nx = steps * (2 * DOF + DOF_A);
        ng = 2 * (steps - 1) * DOF;
        // xl.resize(nx);
        // xu.resize(nx);
        // xi.resize(nx);
        gl.resize(ng);
        gu.resize(ng);
        // initialize gl and gu with zero
        for (int i = 0; i < ng; i++) {
            gl[i] = 0.0;
            gu[i] = 0.0;
        }

        for(int i = 0; i < weight_parameters.size(); i++) {
            std::cout << "OCP - weight_parameters[" << i << "] = " << weight_parameters[i] << std::endl;
        }
    }

    OCP::~OCP() {
    }


    bool OCP::solve(Dvector &xi,
                    Dvector &xu,
                    Dvector &xl,
                    // std::vector<Scalar> &cost_weights_parameters,
                    const std::string &options) {

        // TODO: implement
        CppAD::ipopt::solve<Dvector, FG_eval>(options, xi, xl, xu, gl, gu, fg_eval, solution);
        bool success = solution.status == CppAD::ipopt::solve_result<Dvector>::success;

        return success;
    }



} // ocp
