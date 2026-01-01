#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")

#include <vector>
#include <unordered_set>
#include <map>
#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include "vec_op.hpp"
#include "basics.hpp"
#include <eigen-3.3.8/Eigen/Core>
#include <eigen-3.3.8/Eigen/Dense>
#include <eigen-3.3.8/Eigen/Sparse>
#include <eigen-3.3.8/Eigen/StdVector>
#include <boost/math/special_functions/lambert_w.hpp> // For lambert_w function.

using namespace std;
using namespace Eigen;
using boost::math::lambert_w0;

using ll = long long;
using ull = unsigned long long;
#define all(v) (v).begin(), (v).end()

#define INF 1e12

int dimension = 2;
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Matrix2d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Matrix2i)

Vector2d uniform_distribution_on_sphere(mt19937_64 &mt)
{
    Vector2d vec;
    vec(0) = rand_normal(mt);
    vec(1) = rand_normal(mt);
    double norm = vec.norm();
    vec /= norm;
    return vec;
}

struct cell
{
    Vector2d pos = VectorXd::Zero(dimension);
    double theta;
    double p;
    Vector2d p_vec_std = VectorXd::Zero(dimension);
    double psi;
    double cos_psi;
    double sin_psi;
    Matrix2d RotMat;
    unordered_set<int> neighbors;
};

struct Devo_System
{
    using state = unordered_map<int, cell>;
    int num_cell;
    int space_resolution_cellgrid;
    double system_L;
    double delta_t;
    ll T_MAX_DEVO;
    double interaction_radius_fold;
    double tau_V;
    double A_ij;
    double p_size;
    ll tau_div;
    double tau_B;

    double PSI = 0;
    double delta_x_cellgrid = system_L / space_resolution_cellgrid;
    int num_points_per_cellgrid = space_resolution_cellgrid + 1;

    double D_ij = 1;

    double eq_radius = 1.0;
    double beta = -(2 * eq_radius) / lambert_w0(-exp(-2 * eq_radius) * 2 * eq_radius); // -(2*eq_radius)/W(-exp(-2*eq_radius)*2*eq_radius)
    double interaction_radius = interaction_radius_fold * eq_radius;

    Vector2i pos2lat_cellgrid(const state &x, int i)
    {
        Vector2i lat;
        lat = (x.at(i).pos / delta_x_cellgrid + Vector2d::Constant(0.5)).cast<int>();
        return lat;
    }

    MatrixXd rotation_matrix2D_minus(const double &psi)
    {
        MatrixXd RotMat(dimension, dimension);
        RotMat << sin(psi), cos(psi),
            -cos(psi), sin(psi);
        return RotMat;
    }

    double O_func(const double &angle)
    {
        return sin(angle);
    }

    double O_func_derivative(const double &angle)
    {
        return cos(angle);
    }

    double f(const double &r)
    {
        return exp(-r) - exp(-r / beta);
    }

    double df(const double &r)
    {
        return -exp(-r) + exp(-r / beta) / beta;
    }

    void develop_one_step(state &x, vector<vector<unordered_set<int>>> &which_cell_is_in_which_cellgrid, mt19937_64 &mt, const double &t)
    {
        state dxdt;
        for (auto id : x)
        {
            int i = id.first;
            x.at(i).p_vec_std(0) = cos(x.at(i).theta);
            x.at(i).p_vec_std(1) = sin(x.at(i).theta);
            x.at(i).psi = PSI;
            x.at(i).RotMat = rotation_matrix2D_minus(x.at(i).psi);
        }
        for (auto id : x)
        {
            int i = id.first;
            dxdt[i] = cell{};
            dxdt.at(i).pos = VectorXd::Zero(dimension);
            dxdt.at(i).theta = 0;
            dxdt.at(i).p = 0;

            Vector2d dp_dtheta_std(-sin(x.at(i).theta), cos(x.at(i).theta));
            x.at(i).neighbors.clear();

            Vector2i lat_i = pos2lat_cellgrid(x, i);
            vector<vector<int>> neighbor_grids{{lat_i(0) - 1, lat_i(1) - 1},
                                               {lat_i(0) - 1, lat_i(1)},
                                               {lat_i(0) - 1, lat_i(1) + 1},
                                               {lat_i(0), lat_i(1) - 1},
                                               {lat_i(0), lat_i(1)},
                                               {lat_i(0), lat_i(1) + 1},
                                               {lat_i(0) + 1, lat_i(1) - 1},
                                               {lat_i(0) + 1, lat_i(1)},
                                               {lat_i(0) + 1, lat_i(1) + 1}}; // 8近傍 高速化可能
            for (auto ng : neighbor_grids)
            {
                if (ng[0] < 0 || ng[0] >= num_points_per_cellgrid || ng[1] < 0 || ng[1] >= num_points_per_cellgrid)
                {
                    continue;
                }
                for (auto j : which_cell_is_in_which_cellgrid[ng[0]][ng[1]])
                {
                    if (i == j)
                    {
                        continue;
                    }
                    Vector2d r_vec = x.at(j).pos - x.at(i).pos;
                    double r = r_vec.norm();
                    if (r > interaction_radius) // 恣意性のあるパラメータ、どの距離まで細胞間相互作用が及ぶか
                    {
                        continue;
                    }
                    else
                    {
                        x.at(i).neighbors.insert(j);
                    }
                    Vector2d r_vec_std = r_vec / r;
                    Matrix2d dr_vec_std_dx = -Matrix2d::Identity() / r + r_vec_std * r_vec_std.transpose() / r;

                    double Oi, Oj, S;
                    double sin_theta_i = max(min(outer_product2D(x.at(i).p_vec_std, r_vec_std), 1.0), -1.0), cos_theta_i = max(min(x.at(i).p_vec_std.dot(r_vec_std), 1.0), -1.0), sin_theta_j = max(min(outer_product2D(x.at(j).p_vec_std, r_vec_std), 1.0), -1.0), cos_theta_j = max(min(x.at(j).p_vec_std.dot(r_vec_std), 1.0), -1.0);
                    double angle_r_pi = asin(sin_theta_i), angle_r_pj = asin(sin_theta_j); // -pi/2 ~ pi/2
                    if (cos_theta_i < 0)
                    {
                        if (sin_theta_i > 0)
                        {
                            angle_r_pi = M_PI - angle_r_pi;
                        }
                        else
                        {
                            angle_r_pi = -M_PI - angle_r_pi;
                        }
                    }
                    if (cos_theta_j < 0)
                    {
                        if (sin_theta_j > 0)
                        {
                            angle_r_pj = M_PI - angle_r_pj;
                        }
                        else
                        {
                            angle_r_pj = -M_PI - angle_r_pj;
                        }
                    } // -pi ~ pi
                    Vector2d dS_dx;
                    double dS_dtheta;

                    Oi = x.at(i).p * O_func(angle_r_pi), Oj = x.at(j).p * O_func(angle_r_pj);
                    S = A_ij * Oi * Oj +  D_ij;

                    dS_dx = A_ij * (((Oi * r_vec_std + (x.at(i).RotMat * x.at(i).p * x.at(i).p_vec_std)) / r) / cos_theta_i * O_func_derivative(angle_r_pi) * Oj + ((Oj * r_vec_std + (x.at(j).RotMat * x.at(j).p * x.at(j).p_vec_std)) / r) / cos_theta_j * O_func_derivative(angle_r_pj) * Oi);
                    dS_dtheta = A_ij * x.at(i).p * outer_product2D(dp_dtheta_std, r_vec_std) / cos_theta_i * O_func_derivative(angle_r_pi) * Oj;

                    double f_r = f(r);
                    double df_dr = df(r);
                    dxdt.at(i).pos += -tau_V * (S * df_dr * -r_vec_std + f_r * dS_dx);                     // position
                    dxdt.at(i).theta += -tau_V * (f_r * dS_dtheta) - tau_B * dp_dtheta_std.dot(r_vec_std); // theta
                }
            }
        }

        for (auto id : x)
        {
            int i = id.first;
            Vector2i p_before = pos2lat_cellgrid(x, i);
            x.at(i).pos += (dxdt.at(i).pos) * delta_t;
            x.at(i).pos = x.at(i).pos.cwiseMax(Vector2d::Zero()).cwiseMin(Vector2d::Constant(system_L));
            // register which grid the cell is in
            Vector2i p_after = pos2lat_cellgrid(x, i);
            if (p_before != p_after)
            {
                which_cell_is_in_which_cellgrid[p_before(0)][p_before(1)].erase(i);
                which_cell_is_in_which_cellgrid[p_after(0)][p_after(1)].insert(i);
            }
            x.at(i).theta += (dxdt.at(i).theta) * delta_t;
            if (x.at(i).theta > M_PI)
            {
                while (x.at(i).theta > M_PI)
                {
                    x.at(i).theta -= 2 * M_PI;
                }
            }
            else if (x.at(i).theta < -M_PI)
            {
                while (x.at(i).theta < -M_PI)
                {
                    x.at(i).theta += 2 * M_PI;
                }
            }
            x.at(i).p += dxdt.at(i).p * delta_t;
        }
    }
};

void div_kill(Devo_System &System, Devo_System::state &System_State, vector<vector<unordered_set<int>>> &which_cell_is_in_which_cellgrid, int &cell_id, mt19937_64 &mt)
{
    // choose 1 random cell and divide
    int i = rand() % System_State.size();

    cell daughter_cell = System_State.at(i);

    Vector2d vec = uniform_distribution_on_sphere(mt);
    daughter_cell.pos = System_State.at(i).pos + vec * System.eq_radius * 2;
    daughter_cell.pos = daughter_cell.pos.cwiseMax(Vector2d::Zero()).cwiseMin(Vector2d::Constant(System.system_L));  

    System_State.insert(make_pair(cell_id, daughter_cell));
    Vector2i pos = System.pos2lat_cellgrid(System_State, cell_id);
    which_cell_is_in_which_cellgrid[pos(0)][pos(1)].insert(cell_id);
    cell_id++;
}

void development_record(Devo_System &System, Devo_System::state &System_State, vector<vector<unordered_set<int>>> &which_cell_is_in_which_cellgrid, mt19937_64 &mt, ofstream &redev_record, const ll &t_evo)
{
    int cell_id = System_State.size();
    for (ll t_devo = 0; t_devo < System.T_MAX_DEVO; t_devo++)
    {
        // euler method
        System.develop_one_step(System_State, which_cell_is_in_which_cellgrid, mt, t_devo);
        // divide 1 cell per tau_div steps
        if (t_devo % System.tau_div == 0)
        {
            div_kill(System, System_State, which_cell_is_in_which_cellgrid, cell_id, mt);
        }

        if (t_devo % 100 == 0)
        {
            for (auto id : System_State)
            {
                int i = id.first;
                redev_record << t_evo << " " << t_devo << " " << System_State.size() << " " << i << " ";
                for (int d = 0; d < dimension; d++)
                {
                    redev_record << System_State.at(i).pos(d) << " "; // position
                }
                redev_record << System_State.at(i).theta << " "; // polarity
                redev_record << System_State.at(i).p << " "; // p
                redev_record << endl;
            }
        }
    }
}

int main(int argc, const char **argv)
{
    int seed = stoi(argv[1]);
    int p_size_log = stod(argv[2]);
    int tau_B_log = stod(argv[3]);

    double system_L = 100;
    int space_resolution_cellgrid = 30;
    double delta_t = 0.05;
    ll T_MAX_DEVO = 300000;
    double interaction_radius_fold = 2.5;
    ll tau_div = 1000;
    double tau_V = 10;
    double A_ij = 1;
    int resolution = 5;
    double base = pow(10, 1.0/resolution);
    double p_size = 0.01 * pow(base,p_size_log);
    double tau_B = 0.01 * pow(base,tau_B_log);
    int initial_num_cell = 1;

    string id = "";
    for (int i = 1; i < argc; i++)
    {
        id += argv[i];
        id += "_";
    }

    id += "model_base_2D";
    ofstream redev("../data_model_base_2D/redev" + id + ".txt");

    int num_points_per_dim_cellgrid = space_resolution_cellgrid + 1; 

    size_t seed_t = hash<string>{}(id);
    mt19937_64 mt(seed_t);

    Devo_System::state initial_state{};
    Devo_System System = Devo_System{initial_num_cell, space_resolution_cellgrid, system_L, delta_t, T_MAX_DEVO, interaction_radius_fold, tau_V, A_ij, p_size, tau_div, tau_B};
    vector<vector<unordered_set<int>>> which_cell_in_which_cellgrid(num_points_per_dim_cellgrid, vector<unordered_set<int>>(num_points_per_dim_cellgrid, unordered_set<int>(0)));
    for (int i = 0; i < initial_num_cell; i++)
    {
        initial_state[i] = cell{};
        initial_state.at(i).pos = VectorXd::Constant(dimension, system_L / 2) + VectorXd::Random(dimension);
        initial_state.at(i).theta = rand_real_m11(mt) * M_PI;
        initial_state.at(i).p = p_size;
        Vector2i pos = System.pos2lat_cellgrid(initial_state, i);
        which_cell_in_which_cellgrid[pos(0)][pos(1)].insert(i);
    }
    Devo_System::state System_State = initial_state;

    int t_evo = 0;
    development_record(System, System_State, which_cell_in_which_cellgrid, mt, redev, t_evo);
}
