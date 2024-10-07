//
// Created by gebmer on 22.06.24.
//

#ifndef OCP_H
#define OCP_H

#include "pinocchio/autodiff/cppad.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/rnea.hpp"

namespace robocrane
{
    using CppAD::AD;
    using Eigen::MatrixXd;

    constexpr int DOF_A = 7;
    constexpr int DOF_U = 2;
    constexpr int DOF = DOF_A + DOF_U;

    /* pinocchio types */
    typedef double Scalar;
    using Model = pinocchio::ModelTpl<Scalar>;
    using Data = Model::Data;
    using JointVector = Model::ConfigVectorType;
    using TangentVector = Model::TangentVectorType;
    using ControlVector = Eigen::Vector<double, DOF_A>;

    /* autodiff types */
    typedef CppAD::AD<Scalar> ADScalar;
    using ADModel = pinocchio::ModelTpl<ADScalar>;
    using ADData = ADModel::Data;
    using ADJointVector = ADModel::ConfigVectorType;
    using ADTangentVector = ADModel::TangentVectorType;
    using ADMotion = Model::Motion;
    using VectorXAD = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;
    using ADControlVector = Eigen::Vector<ADScalar, DOF_A>;

    using MatrixXAD = Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using SE3AD = pinocchio::SE3Tpl<ADScalar>;
    using Vector3AD = Eigen::Matrix<ADScalar, 3, 1>;


    /* Define class for function and gradient evaluation for the CppAD Ipopt interface */

    class FG_eval
    {
        int N_split;        
        int N_new = steps_; 

    public:
        typedef CPPAD_TESTVECTOR(ADScalar) ADvector;

        FG_eval(const ADModel &cpin_model,
                const ADData &cpin_data,
                const int steps,
                const Scalar h,
                const int tool_frame_id,
                const std::vector<Scalar> &weight_parameters,
                int &N_split,
                int &N_new)
            : cmodel(cpin_model),
              cdata(cpin_data),
              steps_(steps),
              dof_(DOF),
              dof_a_(DOF_A),
              h_(h),
              tool_frame_id_(tool_frame_id),
              weight_parameters_(weight_parameters),
              N_split(N_split),
              N_new(N_new)
        {

            // print weight parameters
            for (int i = 0; i < weight_parameters.size(); ++i)
            {
                std::cout << "FG_eval - weight_parameters[" << i << "] = " << weight_parameters[i] << std::endl;
            }
        }

        void operator()(ADvector &fg, const ADvector &x) const
        {
            std::cout << "FG_eval - operator()" << std::endl;
            std::cout << "fg.size(): " << fg.size() << std::endl;
            std::cout << "x.size(): " << x.size() << std::endl;

            int num_constraints = 2*(steps_ - 1) * dof_;
            fg.resize(1 + num_constraints);

            MatrixXAD q(steps_, dof_);
            MatrixXAD qp(steps_, dof_);
            MatrixXAD u(steps_, dof_a_);

            int idx = 0;
            for (int i = 0; i < steps_; ++i)
            {
                for (int j = 0; j < dof_; ++j)
                {
                    q(i, j) = x[idx++];
                }
            }
            for (int i = 0; i < steps_; ++i)
            {
                for (int j = 0; j < dof_; ++j)
                {
                    qp(i, j) = x[idx++];
                }
            }
            for (int i = 0; i < steps_; ++i)
            {
                for (int j = 0; j < dof_a_; ++j)
                {
                    u(i, j) = x[idx++];
                }
            }

            ADScalar fq = 0.0;
            ADScalar fqp = 0.0;
            ADScalar fu = 0.0;
            ADScalar fz = 0.0;

            ADScalar kappa = 1.0;      // example value, replace with actual parameter
            ADScalar z_min = 0.0;      // example value, replace with actual parameter
            ADScalar abs_param = 1e-5; // example value, replace with actual parameter

            VectorXAD qd = VectorXAD::Zero(dof_); // example value, replace with actual parameter
            VectorXAD qw = VectorXAD::Zero(dof_); // example value, replace with actual parameter

            VectorXAD cost_weight_parameters(weight_parameters_.size());
            for (int i = 0; i < weight_parameters_.size(); ++i)
            {
                // TODO: cost_weight_parmaters replace with weight_parameters (as reference)?
                cost_weight_parameters[i] = weight_parameters_[i];
            }

            std::cout << "cost_weight_parameters.size(): " << cost_weight_parameters.size() << std::endl;

            MatrixXAD cal_qpp_u(steps_, dof_ - dof_a_);
            cal_qpp_u.setZero();

            for (int i = 0; i < steps_; ++i)
            {
                VectorXAD qpp_a = u.row(i);
                VectorXAD q_i = q.row(i);
                VectorXAD qp_i = qp.row(i);

                // Compute mass matrix and non-linear effects
                pinocchio::crba(cmodel, cdata, q_i);
                pinocchio::nonLinearEffects(cmodel, cdata, q_i, qp_i);

                MatrixXAD M = cdata.M.cast<ADScalar>();
                VectorXAD nle = cdata.nle.cast<ADScalar>();

                MatrixXAD Mu = M.bottomRightCorner(dof_ - dof_a_, dof_ - dof_a_);
                MatrixXAD Mua = M.bottomLeftCorner(dof_ - dof_a_, dof_a_);
                VectorXAD nleu = nle.tail(dof_ - dof_a_);

                auto Mu_inv = Mu.inverse();
                // TODO: missing damping
                VectorXAD qpp_u_tmp = Mu_inv * (-Mua * qpp_a - nleu);
                cal_qpp_u.row(i) = qpp_u_tmp.transpose();
            }

            auto x_concat = MatrixXAD(steps_, 2 * dof_);
            x_concat << q, qp;
            auto qpp = MatrixXAD(steps_, dof_);
            qpp.block(0,0,steps_,dof_a_) = u;
            qpp.block(0,dof_a_,steps_,dof_-dof_a_) = cal_qpp_u;
            auto f_concat = MatrixXAD(steps_, 2 * dof_);
            f_concat << qp, qpp;

            std::vector<ADScalar> g(2*dof_*(steps_-1), 0.0);
            // map g to eigen matrix for easier indexing
            MatrixXAD g_matrix = Eigen::Map<MatrixXAD>(g.data(), 2*dof_, steps_-1);

            for (int i = 0; i < steps_; ++i)
            {
                VectorXAD e = (i >= N_split) ? q.row(i).transpose() - qd : q.row(i).transpose() - qw;

                VectorXAD abs_error = (e.array().square() + abs_param * abs_param).sqrt() - abs_param;
                for (int j = 0; j < dof_; ++j)
                {
                    fq += (i >= N_split)
                              ? cost_weight_parameters[4] * abs_error[j]
                              : cost_weight_parameters[0] * abs_error[j];
                }

                fqp += qp.row(i).tail(dof_ - dof_a_).squaredNorm();

                if (i > 0)
                {
                    // for (int j = 0; j < dof_; ++j)
                    // {
                    //     g.push_back(q(i, j) - q(i - 1, j) - h_ * (qp(i, j) + qp(i - 1, j)) / 2.0);
                    // }
                    g_matrix.col(i-1) = x_concat.row(i).transpose() - x_concat.row(i-1).transpose() - h_ * (f_concat.row(i).transpose() + f_concat.row(i-1).transpose()) / 2.0;
                }
            }

            for (int i = 0; i < steps_ - 1; ++i)
            {
                fu += u.row(i).squaredNorm();
            }

            // Add soft max constraint in z direction
            for (int i = 0; i < steps_; ++i)
            {
                pinocchio::forwardKinematics(cmodel, cdata, q.row(i).transpose(), qp.row(i).transpose(), ADJointVector::Zero(dof_));
                pinocchio::updateFramePlacement(cmodel, cdata, tool_frame_id_);
                SE3AD oMf_tool = cdata.oMf[tool_frame_id_];
                Vector3AD position = oMf_tool.translation();
                ADScalar z = position[2];

                fz += kappa * CppAD::log(1 + CppAD::exp(-kappa * (z - z_min)));
            }

            fg[0] = fq +
                    cost_weight_parameters[1] * fqp +
                    cost_weight_parameters[3] * fu +
                    cost_weight_parameters[5] * fz;

            for (int i = 0; i < g.size(); ++i)
            {
                fg[i + 1] = g[i];
            }
            std::cout << "g.size: " << g.size() << std::endl;
            std::cout << "FG_eval - operator() - end" << std::endl;
        }

    private:
        const ADModel cmodel;
        mutable ADData cdata;
        int steps_;
        int dof_;
        int dof_a_;
        double h_;
        int tool_frame_id_;
        std::vector<double> weight_parameters_;
    };

    class OCP
    {
    public:
        typedef CPPAD_TESTVECTOR(Scalar) Dvector;

    private:
        // int nx;       // # of independent variables (domain dimension for f and g)
        int ng;       // # of constraints (range dimension for g)
        int steps;
        Dvector xi;      // initial value for the independent variables
        Dvector xl, xu;  // lower and upper bounds for the independent variables
        Dvector gl, gu;  // lower and upper bounds for the constraints
        FG_eval fg_eval; // object that computes objective and constraints
        ADModel &cmodel;
        ADData &cdata;
        // MatrixXd g_dyn_ub, g_dyn_lb;
        // std::string options; // string containing options for IPOPT

        CppAD::ipopt::solve_result<Dvector> solution; // place to return solution

    public:
        // OCP();
        OCP(ADModel &cpin_model,
            ADData &cpin_data,
            const int steps,
            const Scalar step_size,
            const int tool_frame_id,
            const std::vector<Scalar> &weight_parameters,
            int &n_split, int &n_new
            // MatrixXd g_dyn_ub, MatrixXd g_dyn_lb
            );
        ~OCP();
        bool solve(Dvector &xi,
                   Dvector &xu,
                   Dvector &xl,
                   // std::vector<Scalar> &cost_weights_parameters,
                   const std::string &options);

        CppAD::ipopt::solve_result<Dvector> get_solution() { return solution; }

    private:
        
    };
} // namespace ocp

#endif // OCP_H
