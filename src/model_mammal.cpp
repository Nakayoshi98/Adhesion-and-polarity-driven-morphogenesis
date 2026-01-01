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
    int num_cell_type;
    Vector2d pos = VectorXd::Zero(dimension);
    double theta;
    double p;
    Vector2d p_vec_std = VectorXd::Zero(dimension);
    double psi;
    double cos_psi;
    double sin_psi;
    Matrix2d RotMat;
    double nut;
    double division_phase;
    unordered_set<int> neighbors;
};

struct Devo_System
{
    using physical_space = VectorXd;
    using state = unordered_map<int, cell>;
    int num_cell;
    int num_cell_type;
    int space_resolution_nutgrid;
    int space_resolution_cellgrid;
    double diffusion_constant;
    double system_L;
    double uptake_constant;
    double supply_rate;
    double linear_consumption_constant;
    double delta_t;
    ll T_MAX_DEVO;
    double noise_size_pos;
    double noise_size_theta;
    double interaction_radius_fold;
    double mu;
    double tau_V;
    double nut_min_threshold;
    double nut_max_threshold;
    double A_ij;
    double C_ij;
    double D_ij;
    double communication_constant;
    double gain_p;
    double threshold_p;
    int neighbor_size;
    double nut_phys_radius_ratio;
    double tau_p;
    double max_uptake;
    double tau_div;
    double tau_B;
    double time_deviation;

    double PSI = 0;
    double potential_force = 1;
    double delta_x_nutgrid = system_L / space_resolution_nutgrid;
    double delta_x_cellgrid = system_L / space_resolution_cellgrid;
    double nu = diffusion_constant / pow(delta_x_nutgrid, 2);
    int num_points_per_cellgrid = space_resolution_cellgrid + 1;
    int num_points_per_nutgrid = space_resolution_nutgrid + 1;
    ll num_points_in_space_nutgrid = powint(num_points_per_nutgrid, dimension);

    double eq_radius = 1.001 * delta_x_nutgrid;
    double beta = -(2 * eq_radius) / lambert_w0(-exp(-2 * eq_radius) * 2 * eq_radius); // -(2*eq_radius)/W(-exp(-2*eq_radius)*2*eq_radius)
    double interaction_radius = interaction_radius_fold * eq_radius;
    double nut_radius = nut_phys_radius_ratio * eq_radius;

    Vector2i pos2lat_cellgrid(const state &x, int i)
    {
        Vector2i lat;
        lat = (x.at(i).pos / delta_x_cellgrid + Vector2d::Constant(0.5)).cast<int>();
        return lat;
    }

    Vector2i pos2lat_nutgrid(const state &x, int i)
    {
        Vector2i lat;
        lat = (x.at(i).pos / delta_x_nutgrid + Vector2d::Constant(0.5)).cast<int>();
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

    Vector2d Fx(const Vector2d &r_vec) // cell_potential_field_excluded_volume_effect
    {
        double r = r_vec.norm();
        if (r < eq_radius)
        {
            return r_vec / r * potential_force;
        }
        else
        {
            return Vector2d::Zero();
        }
    }

    double f(const double &r) // spring potential
    {
        return exp(-r) - exp(-r / beta);
    }

    double df(const double &r)
    {
        return -exp(-r) + exp(-r / beta) / beta;
    }

    double nut2pol(const double &nut)
    {
        return sigmoid(nut - threshold_p, gain_p);
    }

    void develop_one_step(state &x, physical_space &ps, vector<vector<unordered_set<int>>> &which_cell_is_in_which_cellgrid, SparseMatrix<double> &A, mt19937_64 &mt, const double &t)
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
            dxdt[i] = cell{num_cell_type};
            dxdt.at(i).nut = 0;
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
                    double sin_theta_i = outer_product2D(x.at(i).p_vec_std, r_vec_std), cos_theta_i = x.at(i).p_vec_std.dot(r_vec_std), sin_theta_j = outer_product2D(x.at(j).p_vec_std, r_vec_std), cos_theta_j = x.at(j).p_vec_std.dot(r_vec_std);
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
                    S = A_ij * Oi * Oj + C_ij * (1 - x.at(i).p) * (1 - x.at(j).p) + D_ij;

                    dS_dx = A_ij * (((x.at(i).p * sin_theta_i * r_vec_std + (x.at(i).RotMat * x.at(i).p * x.at(i).p_vec_std)) / r) / cos_theta_i * O_func_derivative(angle_r_pi) * Oj + ((x.at(j).p * sin_theta_j * r_vec_std + (x.at(j).RotMat * x.at(j).p * x.at(j).p_vec_std)) / r) / cos_theta_j * O_func_derivative(angle_r_pj) * Oi);
                    dS_dtheta = A_ij * x.at(i).p * outer_product2D(dp_dtheta_std, r_vec_std) / cos_theta_i * O_func_derivative(angle_r_pi) * Oj;

                    double f_r = f(r);
                    double df_dr = df(r);
                    dxdt.at(i).pos += -tau_V * (S * df_dr * -r_vec_std + f_r * dS_dx); // position
                    // dxdt.at(i).theta += -tau_V * (f_r * dS_dtheta);                    // theta
                    dxdt.at(i).theta += -tau_V * (f_r * dS_dtheta) - tau_B * dp_dtheta_std.dot(r_vec_std); // theta

                    dxdt.at(i).nut += communication_constant * (x.at(j).nut - x.at(i).nut) / neighbor_grids.size(); // nutrient communication
                }
            }
        }
        physical_space x_eff = ps;
        unordered_set<ll> no_diffusion_index_list{};
        auto B = A;
        for (auto id : x)
        {
            int i = id.first;
            // calculate the effective concentration of nutrient by adding the exclusive effect of cells to the actual concentration
            Vector2i p = pos2lat_nutgrid(x, i);
            double nut_io = 0;

            int out_of_cell_gridpoints = 0;
            for (int n = 0; n < neighbor_size * neighbor_size; n++)
            {
                Vector2i neighbor_nutgrid = Vector2i(p(0) + n % neighbor_size - 2, p(1) + n / neighbor_size - 2);
                Vector2d vec2neighbor_nutgrid = neighbor_nutgrid.cast<double>() * delta_x_nutgrid - x.at(i).pos;
                double r = vec2neighbor_nutgrid.norm();
                if (r > nut_radius)
                {
                    out_of_cell_gridpoints++;
                }
            }

            for (int n = 0; n < neighbor_size * neighbor_size; n++)
            {
                Vector2i neighbor_nutgrid = Vector2i(p(0) + n % neighbor_size - 2, p(1) + n / neighbor_size - 2);
                Vector2d vec2neighbor_nutgrid = neighbor_nutgrid.cast<double>() * delta_x_nutgrid - x.at(i).pos;

                if (neighbor_nutgrid[0] < 0 || neighbor_nutgrid[0] >= num_points_per_nutgrid || neighbor_nutgrid[1] < 0 || neighbor_nutgrid[1] >= num_points_per_nutgrid) // out of the boundary
                {
                    continue;
                }

                int row = neighbor_nutgrid(0), col = neighbor_nutgrid(1);
                ll index = row * num_points_per_nutgrid + col;

                double r = vec2neighbor_nutgrid.norm();
                Vector2d r_vec_std = vec2neighbor_nutgrid / r;

                if (r < nut_radius)
                {
                    ps(index) = 0;
                    no_diffusion_index_list.insert(index);
                }
                else
                {
                    double uptake = uptake_constant * min(max(x_eff(index) - x.at(i).nut, 0.), max_uptake) / out_of_cell_gridpoints; // ここ漏れ出しなし
                    nut_io += uptake;
                    ps(index) -= uptake * delta_t;
                }
            }
            dxdt.at(i).nut += nut_io - linear_consumption_constant * x.at(i).nut; // nutrient consumption
            dxdt.at(i).p = tau_p * (nut2pol(x.at(i).nut) - x.at(i).p);
            dxdt.at(i).division_phase = tau_div;
        }

        for (ll index : no_diffusion_index_list)
        {
            ll i = index / num_points_per_nutgrid, j = index % num_points_per_nutgrid;
            if (i < num_points_per_nutgrid - 1)
            {
                A.coeffRef(index + num_points_per_nutgrid, index) = 0;
                A.coeffRef(index, index + num_points_per_nutgrid) = 0;
                A.coeffRef(index + num_points_per_nutgrid, index + num_points_per_nutgrid) += nu;
            }
            if (i > 0)
            {
                A.coeffRef(index - num_points_per_nutgrid, i) = 0;
                A.coeffRef(index, index - num_points_per_nutgrid) = 0;
                A.coeffRef(index - num_points_per_nutgrid, index - num_points_per_nutgrid) += nu;
            }
            if (j < num_points_per_nutgrid - 1)
            {
                A.coeffRef(index + 1, index) = 0;
                A.coeffRef(index, index + 1) = 0;
                A.coeffRef(index + 1, index + 1) += nu;
            }
            if (j > 0)
            {
                A.coeffRef(index - 1, index) = 0;
                A.coeffRef(index, index - 1) = 0;
                A.coeffRef(index - 1, index - 1) += nu;
            }
        }
        for (ll index : no_diffusion_index_list)
        {
            A.coeffRef(index, index) = 0;
        }
        A.prune(0.0); // no diffusion

        ps += (A * ps) * delta_t; // diffusion 高速化可能
        // calculate the change of nutrient concentration
        for (int i = 0; i < num_points_per_nutgrid; i++)
        {
            int j = 0;
            ll index;
            index = i * num_points_per_nutgrid + j;
            ps(index) += supply_rate * delta_t;
            j = num_points_per_nutgrid - 1;
            index = i * num_points_per_nutgrid + j;
            ps(index) += supply_rate * delta_t;
        }
        for (int j = 0; j < num_points_per_nutgrid; j++)
        {
            int i = 0;
            ll index;
            index = i * num_points_per_nutgrid + j;
            ps(index) += supply_rate * delta_t;
            i = num_points_per_nutgrid - 1;
            index = i * num_points_per_nutgrid + j;
            ps(index) += supply_rate * delta_t;
        }

        // // reset A
        A = B;

        for (auto id : x)
        {
            int i = id.first;
            Vector2i p_before = pos2lat_cellgrid(x, i);
            Vector2d noise_pos = rand_normal(mt) * noise_size_pos * uniform_distribution_on_sphere(mt);
            x.at(i).pos += (dxdt.at(i).pos + noise_pos) * delta_t;
            x.at(i).pos = x.at(i).pos.cwiseMax(Vector2d::Zero()).cwiseMin(Vector2d::Constant(system_L));
            // register which grid the cell is in
            Vector2i p_after = pos2lat_cellgrid(x, i);
            if (p_before != p_after)
            {
                which_cell_is_in_which_cellgrid[p_before(0)][p_before(1)].erase(i);
                which_cell_is_in_which_cellgrid[p_after(0)][p_after(1)].insert(i);
            }
            double noise_theta = rand_normal(mt) * noise_size_theta;
            x.at(i).theta += (dxdt.at(i).theta + noise_theta) * delta_t;
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
            x.at(i).nut += dxdt.at(i).nut * delta_t;
            x.at(i).nut = x.at(i).nut < 0 ? 0 : x.at(i).nut; // nutrientが負になることがあるので
            x.at(i).p += dxdt.at(i).p * delta_t;
            x.at(i).division_phase += dxdt.at(i).division_phase * (1 + rand_normal(mt) * time_deviation) * delta_t;
        }
    }
};

void div_kill(Devo_System &System, Devo_System::state &System_State, Devo_System::physical_space &physical_space, vector<vector<unordered_set<int>>> &which_cell_is_in_which_cellgrid, int &cell_id, mt19937_64 &mt)
{
    vector<int> kill_list(0);
    vector<cell> daughter_list(0);
    for (auto id : System_State)
    {
        int i = id.first;

        if (System_State.at(i).division_phase > 1)
        {
            System_State.at(i).division_phase = 0;
            cell daughter_cell = System_State.at(i);

            Vector2d vec = uniform_distribution_on_sphere(mt);
            daughter_cell.pos = System_State.at(i).pos + vec * System.eq_radius * 2;
            daughter_cell.pos = daughter_cell.pos.cwiseMax(Vector2d::Zero()).cwiseMin(Vector2d::Constant(System.system_L));
            daughter_list.push_back(daughter_cell);
        }
    }
    for (cell daughter_cell : daughter_list)
    {
        System_State.insert(make_pair(cell_id, daughter_cell));
        Vector2i pos = System.pos2lat_cellgrid(System_State, cell_id);
        which_cell_is_in_which_cellgrid[pos(0)][pos(1)].insert(cell_id);
        cell_id++;
    }
}

void development_record(Devo_System &System, Devo_System::state &System_State, Devo_System::physical_space &physical_space, vector<vector<unordered_set<int>>> &which_cell_is_in_which_cellgrid, SparseMatrix<double> &&A, mt19937_64 &mt, ofstream &redev_record, ofstream &redev_nut_field, const ll &t_evo)
{
    int cell_id = System_State.size();
    for (ll t_devo = 0; t_devo < System.T_MAX_DEVO; t_devo++)
    {
        // euler method
        System.develop_one_step(System_State, physical_space, which_cell_is_in_which_cellgrid, A, mt, t_devo);
        // check nutrient condition of the cells & kill or divide
        div_kill(System, System_State, physical_space, which_cell_is_in_which_cellgrid, cell_id, mt);

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
                redev_record << System_State.at(i).p << " ";   // p
                redev_record << System_State.at(i).nut << " "; // nutrient
                redev_record << endl;
            }
            for (int i = 0; i < System.num_points_per_nutgrid; i++)
            {
                for (int j = 0; j < System.num_points_per_nutgrid; j++)
                {
                    if (physical_space(i * System.num_points_per_nutgrid + j) > 1e6)
                    {
                        exit(0);
                    }
                    redev_nut_field << physical_space(i * System.num_points_per_nutgrid + j) << " ";
                }
                redev_nut_field << endl;
            }
        }
    }
}

int main(int argc, const char **argv)
{
    double diffusion_constant = 20;
    double uptake_constant = 0.07;
    double initial_nutrient_ps = 2.0;
    double initial_nutrient_cell = 2.0;
    double supply_rate = 0.5;
    double linear_consumption_constant = 0.007;
    int num_cell_type = 1;
    double system_L = 100;
    int space_resolution_nutgrid = 101;
    int space_resolution_cellgrid = 1;
    double delta_t = 0.005;
    ll T_MAX_DEVO = 40000;
    double noise_size_pos = 1.0;
    double noise_size_theta = 0;
    double interaction_radius_fold = 2.5;
    int seed = 4;
    double mu = -1;
    double tau_V = 10;
    double nut_min_threshold = 0.1;
    double nut_max_threshold = 1.5;
    double A_ij = 1;
    double C_ij = 2;
    double D_ij = 1;
    double communication_constant = 0);
    double gain_p = 100;
    double threshold_p = 0.5;
    int neighbor_size = 5;
    double nut_phys_radius_ratio = 1.5;
    double tau_p = 5;
    double max_uptake = 10;
    double tau_div = 0.05;
    double tau_B = 2;
    double time_deviation = 10;

    int initial_num_cell = 1;

    string id = "";
    for (int i = 1; i < argc; i++)
    {
        id += argv[i];
        id += "_";
    }

    id += "model2";

    ofstream redev("../data2D_model2/redev" + id + ".txt");
    ofstream redev_nut_field("../data2D_model2/redev_nut_field" + id + ".txt");

    int num_points_per_dim_nutgrid = space_resolution_nutgrid + 1;   // 植木算
    int num_points_per_dim_cellgrid = space_resolution_cellgrid + 1; // 植木算
    ll num_points_in_space_nutgrid = powint(num_points_per_dim_nutgrid, dimension);
    double delta_x_nutgrid = system_L / space_resolution_nutgrid;

    vector<Triplet<double>> triplets;
    double nu = diffusion_constant / pow(delta_x_nutgrid, 2);
    // 2D diffusion with Neumann boundary condition
    for (int i = 0; i < num_points_per_dim_nutgrid; i++)
    {
        for (int j = 0; j < num_points_per_dim_nutgrid; j++)
        {

            ll index = i * num_points_per_dim_nutgrid + j, flows = 0;
            if (i > 0)
            {
                triplets.emplace_back(index, index - num_points_per_dim_nutgrid, nu);
                flows++;
            }
            if (i < num_points_per_dim_nutgrid - 1)
            {
                triplets.emplace_back(index, index + num_points_per_dim_nutgrid, nu);
                flows++;
            }
            if (j > 0)
            {
                triplets.emplace_back(index, index - 1, nu);
                flows++;
            }
            if (j < num_points_per_dim_nutgrid - 1)
            {
                triplets.emplace_back(index, index + 1, nu);
                flows++;
            }
            triplets.emplace_back(index, index, -flows * nu);
        }
    }
    // 疎行列の作成
    SparseMatrix<double, RowMajor, int64_t> A(num_points_in_space_nutgrid, num_points_in_space_nutgrid);
    A.setFromTriplets(triplets.begin(), triplets.end());

    // mt19937_64 mt(seed);
    size_t seed_t = hash<string>{}(id);
    mt19937_64 mt(seed_t);
    
    Devo_System::state initial_state{};
    Devo_System::physical_space initial_physical_space = VectorXd::Constant(num_points_in_space_nutgrid, initial_nutrient_ps);

    Devo_System System = Devo_System{initial_num_cell, num_cell_type, space_resolution_nutgrid, space_resolution_cellgrid, diffusion_constant, system_L, uptake_constant, supply_rate, linear_consumption_constant, delta_t, T_MAX_DEVO, noise_size_pos, noise_size_theta, interaction_radius_fold, mu, tau_V, nut_min_threshold, nut_max_threshold, A_ij, C_ij, D_ij, communication_constant, gain_p, threshold_p, neighbor_size, nut_phys_radius_ratio, tau_p, max_uptake, tau_div, tau_B, time_deviation};

    Devo_System::physical_space physical_space = initial_physical_space;
    vector<vector<unordered_set<int>>> which_cell_in_which_cellgrid(num_points_per_dim_cellgrid, vector<unordered_set<int>>(num_points_per_dim_cellgrid, unordered_set<int>(0)));
    for (int i = 0; i < initial_num_cell; i++)
    {
        initial_state[i] = cell{num_cell_type};
        initial_state.at(i).pos = VectorXd::Constant(dimension, system_L / 2) + VectorXd::Random(dimension);
        initial_state.at(i).theta = rand_real_m11(mt) * M_PI;
        initial_state.at(i).nut = initial_nutrient_cell; // initial nutrient condition of the first cell
        initial_state.at(i).p = 0;
        initial_state.at(i).division_phase = 0;
        Vector2i pos = System.pos2lat_cellgrid(initial_state, i);
        which_cell_in_which_cellgrid[pos(0)][pos(1)].insert(i);
    }
    Devo_System::state System_State = initial_state;

    redev_nut_field << fixed << setprecision(6);
    for (int i = 0; i < num_points_per_dim_nutgrid; i++)
    {
        for (int j = 0; j < num_points_per_dim_nutgrid; j++)
        {
            redev_nut_field << physical_space(i * num_points_per_dim_nutgrid + j) << " ";
        }
        redev_nut_field << endl;
    }
    int t_evo = 0;
    development_record(System, System_State, physical_space, which_cell_in_which_cellgrid, A, mt, redev, redev_nut_field, t_evo);
}
