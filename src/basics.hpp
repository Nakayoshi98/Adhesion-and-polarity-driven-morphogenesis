
#include <vector>
#include <random>
#include <eigen-3.3.8/Eigen/Core>
using namespace std;
using namespace Eigen;

normal_distribution<double> rand_normal(0, 1);
uniform_real_distribution<double> rand_real_01(0, 1), rand_real_m11(-1, 1);

long long powint(int x, int y)
{
    long long ans = 1;
    for (int i = 0; i < y; i++)
    {
        ans *= x;
    }
    return ans;
}

double dot(const vector<double> &a, const vector<double> &b)
{
    double ans = 0;
    for (int i = 0; i < a.size(); i++)
    {
        ans += a[i] * b[i];
    }
    return ans;
}

vector<double> dot_Mx(const vector<vector<double>> &M, const vector<double> &x)
{
    vector<double> ans(M.size());
    for (int i = 0; i < M.size(); i++)
    {
        ans[i] = dot(M[i], x);
    }
    return ans;
}

vector<double> outer_product(const vector<double> &a, const vector<double> &b)
{
    int dimension = a.size();
    vector<double> ans(dimension);
    for (int i = 0; i < dimension; i++)
    {
        ans[i] = a[(i + 1) % dimension] * b[(i + 2) % dimension] - a[(i + 2) % dimension] * b[(i + 1) % dimension];
    }
    return ans;
}

// double outer_product2D(const vector<double> &a, const vector<double> &b)
// {
//     return a[0] * b[1] - a[1] * b[0];
// }
double outer_product2D(const Vector2d &a, const Vector2d &b)
{
    return a(0) * b(1) - a(1) * b(0);
}

double norm(const vector<double> &a)
{
    double ans = 0;
    for (int i = 0; i < a.size(); i++)
    {
        ans += a[i] * a[i];
    }
    return sqrt(ans);
}

double L1_abs_sum(const vector<double> &a)
{
    double ans = 0;
    for (int i = 0; i < a.size(); i++)
    {
        ans += abs(a[i]);
    }
    return ans;
}

double angle(const vector<double> &a, const vector<double> &b)
{
    return acos(dot(a, b) / (norm(a) * norm(b)));
}

double sigmoid(double x, double gain)
{
    return 1. / (1. + exp(-gain * x));
}

double sigmoid_derivative(double x, double gain)
{
    return gain * sigmoid(x, gain) * (1. - sigmoid(x, gain));
}

double entropy(const vector<double> &p)
{
    double ans = 0;
    for (int i = 0; i < p.size(); i++)
    {
        if (p[i] > 0)
        {
            ans += p[i] * log2(p[i]);
        }
        else
        {
            ans += 0;
        }
        // ans += p[i] * log2(p[i]);
    }
    return -ans;
}