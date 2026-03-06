#include <RcppArmadillo.h>
#include <roptim.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <unordered_map>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(roptim)]]

using namespace arma;
using namespace Rcpp;
using namespace roptim;

// Link function implementations
namespace links
{
    // Link function type definitions
    using LinkFunction = std::function<arma::vec(const arma::vec &)>;

    // Link function registry
    struct LinkFunctions
    {
        LinkFunction linkinv;
        LinkFunction mu_eta;
    };

    // Logit link
    inline arma::vec logit_linkinv(const arma::vec &eta)
    {
        return 1.0 / (1.0 + exp(-eta));
    }

    inline arma::vec logit_mu_eta(const arma::vec &eta)
    {
        arma::vec mu = logit_linkinv(eta);
        return mu % (1.0 - mu);
    }

    // Probit link
    inline arma::vec probit_linkinv(const arma::vec &eta)
    {
        arma::vec result(eta.n_elem);
        for (size_t i = 0; i < eta.n_elem; i++)
        {
            result(i) = R::pnorm(eta(i), 0.0, 1.0, 1, 0);
        }
        return result;
    }

    inline arma::vec probit_mu_eta(const arma::vec &eta)
    {
        arma::vec result(eta.n_elem);
        for (size_t i = 0; i < eta.n_elem; i++)
        {
            result(i) = R::dnorm(eta(i), 0.0, 1.0, 0);
        }
        return result;
    }

    // Complementary log-log link
    inline arma::vec cloglog_linkinv(const arma::vec &eta)
    {
        return 1.0 - exp(-exp(eta));
    }

    inline arma::vec cloglog_mu_eta(const arma::vec &eta)
    {
        return exp(eta - exp(eta));
    }

    // Cauchit link
    inline arma::vec cauchit_linkinv(const arma::vec &eta)
    {
        arma::vec result(eta.n_elem);
        for (size_t i = 0; i < eta.n_elem; i++)
        {
            result(i) = R::pcauchy(eta(i), 0.0, 1.0, 1, 0);
        }
        return result;
    }

    inline arma::vec cauchit_mu_eta(const arma::vec &eta)
    {
        arma::vec result(eta.n_elem);
        for (size_t i = 0; i < eta.n_elem; i++)
        {
            result(i) = R::dcauchy(eta(i), 0.0, 1.0, 0);
        }
        return result;
    }

    // Log link
    inline arma::vec log_linkinv(const arma::vec &eta)
    {
        return exp(eta);
    }

    inline arma::vec log_mu_eta(const arma::vec &eta)
    {
        return exp(eta);
    }

    // Link function registry
    static const std::unordered_map<std::string, LinkFunctions> link_registry = {
        {"logit", {logit_linkinv, logit_mu_eta}},
        {"probit", {probit_linkinv, probit_mu_eta}},
        {"cloglog", {cloglog_linkinv, cloglog_mu_eta}},
        {"cauchit", {cauchit_linkinv, cauchit_mu_eta}},
        {"log", {log_linkinv, log_mu_eta}}};

    // Get link function by name
    inline LinkFunction get_linkinv(const std::string &link)
    {
        auto it = link_registry.find(link);
        if (it != link_registry.end())
        {
            return it->second.linkinv;
        }
        Rcpp::stop("Unknown link function: " + link);
        return nullptr;
    }

    inline LinkFunction get_mu_eta(const std::string &link)
    {
        auto it = link_registry.find(link);
        if (it != link_registry.end())
        {
            return it->second.mu_eta;
        }
        Rcpp::stop("Unknown link function: " + link);
        return nullptr;
    }
}

// Numerically stable log(1 - exp(x)) for x < 0
// Uses log(-expm1(x)) when x is close to 0 (x > -ln2)
// and log1p(-exp(x)) when x is very negative (x <= -ln2)
inline arma::vec log1mexp(const arma::vec &x)
{
    arma::vec result(x.n_elem);
    // Split into two branches for vectorization
    arma::uvec small = find(x > -M_LN2);  // close to 0: use expm1
    arma::uvec large = find(x <= -M_LN2); // very negative: use exp

    if (small.n_elem > 0)
    {
        // For each element in 'small', compute log(-expm1(x))
        arma::vec xs = x.elem(small);
        for (size_t i = 0; i < xs.n_elem; i++)
        {
            xs(i) = std::log(-std::expm1(xs(i)));
        }
        result.elem(small) = xs;
    }
    if (large.n_elem > 0)
    {
        // Vectorized path: log1p(-exp(x)) for the common case
        result.elem(large) = log1p(-exp(x.elem(large)));
    }
    return result;
}

// Base class for all likelihood functors
class LikelihoodFunctor : public Functor
{
protected:
    const arma::vec &Y;
    const arma::mat &X;
    const arma::vec &offset;
    const arma::vec &weights;

    // Cached indicator vectors for Y=0 and Y>0
    arma::uvec Y0;
    arma::uvec Y1;

    // Cached subsets (constant across all iterations)
    arma::mat X_pos;       // X.rows(Y1)
    arma::mat X_zero;      // X.rows(Y0)
    arma::vec offset_pos;  // offset.elem(Y1)
    arma::vec offset_zero; // offset.elem(Y0)
    arma::vec w_pos;       // weights.elem(Y1)
    arma::vec w_zero;      // weights.elem(Y0)
    arma::vec Y_pos;       // Y.elem(Y1)

public:
    LikelihoodFunctor(const arma::vec &y, const arma::mat &x, const arma::vec &offs, const arma::vec &w)
        : Y(y), X(x), offset(offs), weights(w)
    {
        // Pre-compute indicator vectors
        Y0 = find(Y <= 0);
        Y1 = find(Y > 0);

        // Cache constant subsets to avoid repeated allocation
        if (Y1.n_elem > 0)
        {
            X_pos = X.rows(Y1);
            offset_pos = offset.elem(Y1);
            w_pos = weights.elem(Y1);
            Y_pos = Y.elem(Y1);
        }
        if (Y0.n_elem > 0)
        {
            X_zero = X.rows(Y0);
            offset_zero = offset.elem(Y0);
            w_zero = weights.elem(Y0);
        }
    }

    // Common utility functions
    arma::vec calculate_eta(const arma::vec &parms) const
    {
        return X * parms + offset;
    }

    arma::vec calculate_mu(const arma::vec &eta) const
    {
        return exp(eta);
    }

    // Virtual destructor for proper cleanup in derived classes
    virtual ~LikelihoodFunctor() = default;
};

// Count model Poisson functor
class CountPoissonFunctor : public LikelihoodFunctor
{
private:
    arma::vec lgamma_y_1_cached; // lgamma(Y_pos + 1), constant across iterations

public:
    CountPoissonFunctor(const arma::vec &y, const arma::mat &x, const arma::vec &offs, const arma::vec &w)
        : LikelihoodFunctor(y, x, offs, w)
    {
        if (Y1.n_elem > 0)
        {
            lgamma_y_1_cached.set_size(Y1.n_elem);
            for (size_t i = 0; i < Y1.n_elem; i++)
            {
                lgamma_y_1_cached(i) = lgamma(Y_pos(i) + 1.0);
            }
        }
    }

    double operator()(const arma::vec &parms) override
    {
        // If no Y>0 observations, return 0
        if (Y1.n_elem == 0)
        {
            return 0.0;
        }

        // Calculate mu = exp(X * parms + offset) for Y>0 observations
        arma::vec mu = calculate_mu(X_pos * parms + offset_pos);

        // Calculate log probability of zero: loglik0 = -mu
        arma::vec loglik0 = -mu;

        arma::vec loglik1 = Y_pos % log(mu) - mu - lgamma_y_1_cached;

        // Calculate log-likelihood
        double loglik = arma::dot(w_pos, loglik1) - arma::dot(w_pos, log1mexp(loglik0));

        // Return negative log-likelihood for minimization
        return -loglik;
    }

    void Gradient(const arma::vec &parms, arma::vec &grad) override
    {
        // If no Y>0 observations, return zero gradient
        if (Y1.n_elem == 0)
        {
            grad = arma::zeros<arma::vec>(parms.n_elem);
            return;
        }

        // Calculate eta = X * parms + offset for Y>0 observations
        arma::vec eta = X_pos * parms + offset_pos;

        // Calculate mu = exp(eta)
        arma::vec mu = calculate_mu(eta);

        // Vectorized gradient calculation
        arma::vec loglik0 = -mu; // log probability of zero
        arma::vec grad_term = Y_pos - mu - exp(loglik0 - log1mexp(loglik0) + eta);

        // Single matrix multiplication instead of loop
        grad = X_pos.t() * (w_pos % grad_term);

        // Return negative gradient for minimization
        grad = -grad;
    }
};

// Count model Negative Binomial functor
class CountNegBinFunctor : public LikelihoodFunctor
{
private:
    arma::vec lgamma_y_1_cached; // lgamma(Y_pos + 1), constant across iterations

    // Unique-value lookup for Y_pos: compute expensive special functions
    // (lgamma, digamma) only for distinct Y values, then scatter back.
    arma::vec unique_y_vals;  // sorted unique values of Y_pos
    arma::uvec y_index_map;   // maps each Y_pos element to its index in unique_y_vals

    // Cached intermediates shared between operator() and Gradient().
    // We store the parameter vector used to compute them and reuse
    // the cache only when Gradient() is called with identical parameters.
    arma::vec cached_parms;
    arma::vec cached_eta;
    arma::vec cached_mu;
    double cached_theta;
    arma::vec cached_loglik0;
    arma::vec cached_logratio;
    bool cache_valid;

    void compute_intermediates(const arma::vec &parms)
    {
        int kx = X.n_cols;
        cached_eta = X_pos * parms.subvec(0, kx - 1) + offset_pos;
        cached_mu = exp(cached_eta);
        cached_theta = exp(parms(kx));
        double log_theta = log(cached_theta);
        cached_loglik0 = cached_theta * log_theta - cached_theta * log(cached_theta + cached_mu);
        cached_logratio = cached_loglik0 - log1mexp(cached_loglik0);
        cached_parms = parms;
        cache_valid = true;
    }

public:
    CountNegBinFunctor(const arma::vec &y, const arma::mat &x, const arma::vec &offs, const arma::vec &w)
        : LikelihoodFunctor(y, x, offs, w), cached_theta(0.0), cache_valid(false)
    {
        if (Y1.n_elem > 0)
        {
            // Build unique-value lookup for Y_pos
            unique_y_vals = arma::unique(Y_pos); // returns sorted
            y_index_map.set_size(Y_pos.n_elem);
            for (size_t i = 0; i < Y_pos.n_elem; i++)
            {
                // Binary search since unique_y_vals is sorted
                auto it = std::lower_bound(unique_y_vals.begin(), unique_y_vals.end(), Y_pos(i));
                y_index_map(i) = static_cast<arma::uword>(it - unique_y_vals.begin());
            }

            // Cache lgamma(Y_pos + 1) using unique-value lookup
            arma::vec lgamma_unique(unique_y_vals.n_elem);
            for (size_t j = 0; j < unique_y_vals.n_elem; j++)
            {
                lgamma_unique(j) = lgamma(unique_y_vals(j) + 1.0);
            }
            lgamma_y_1_cached = lgamma_unique.elem(y_index_map);
        }
    }

    double operator()(const arma::vec &parms) override
    {
        // If no Y>0 observations, return 0
        if (Y1.n_elem == 0)
        {
            return 0.0;
        }

        // Compute and cache intermediates
        compute_intermediates(parms);

        // Compute lgamma(Y + theta) only for unique Y values, then scatter
        arma::vec lgamma_unique(unique_y_vals.n_elem);
        for (size_t j = 0; j < unique_y_vals.n_elem; j++)
        {
            lgamma_unique(j) = lgamma(unique_y_vals(j) + cached_theta);
        }
        arma::vec lgamma_y_theta = lgamma_unique.elem(y_index_map);

        // Vectorized negative binomial log probability
        arma::vec loglik1 = lgamma_y_theta - lgamma(cached_theta) - lgamma_y_1_cached +
                            Y_pos % log(cached_mu) + cached_theta * log(cached_theta) -
                            (Y_pos + cached_theta) % log(cached_mu + cached_theta);

        // Calculate log-likelihood
        double loglik = arma::dot(w_pos, loglik1) - arma::dot(w_pos, log1mexp(cached_loglik0));

        // Return negative log-likelihood for minimization
        return -loglik;
    }

    void Gradient(const arma::vec &parms, arma::vec &grad) override
    {
        // Get number of parameters
        int kx = X.n_cols;

        // If no Y>0 observations, return zero gradient
        if (Y1.n_elem == 0)
        {
            grad = arma::zeros<arma::vec>(parms.n_elem);
            return;
        }

        // Reuse cached intermediates from operator() if parameters match
        if (!cache_valid || parms.n_elem != cached_parms.n_elem ||
            !arma::all(parms == cached_parms))
        {
            compute_intermediates(parms);
        }

        // Use cached values
        const arma::vec &eta = cached_eta;
        const arma::vec &mu = cached_mu;
        double theta = cached_theta;
        const arma::vec &logratio = cached_logratio;

        // Vectorized gradient calculation for beta parameters
        arma::vec mu_plus_theta = mu + theta;
        arma::vec log_mu_plus_theta = log(mu_plus_theta);

        arma::vec grad_term = Y_pos - mu % (Y_pos + theta) / mu_plus_theta -
                              exp(logratio + log(theta) - log_mu_plus_theta + eta);

        // Single matrix multiplication instead of loop
        arma::vec grad_beta = X_pos.t() * (w_pos % grad_term);

        // Compute digamma(Y + theta) only for unique Y values, then scatter
        arma::vec digamma_unique(unique_y_vals.n_elem);
        for (size_t j = 0; j < unique_y_vals.n_elem; j++)
        {
            digamma_unique(j) = R::digamma(unique_y_vals(j) + theta);
        }
        arma::vec digamma_y_theta = digamma_unique.elem(y_index_map);

        double digamma_theta = R::digamma(theta);

        // Vectorized first term for grad_logtheta
        arma::vec term3 = digamma_y_theta - digamma_theta + log(theta) - log_mu_plus_theta +
                          1.0 - (Y_pos + theta) / mu_plus_theta;

        // Vectorized second term for grad_logtheta
        arma::vec term4 = exp(logratio) % (log(theta) - log_mu_plus_theta + 1.0 - theta / mu_plus_theta);

        // Sum with weights
        double grad_logtheta = theta * arma::dot(w_pos, term3 + term4);

        // Combine gradients
        grad = arma::zeros<arma::vec>(parms.n_elem);
        grad.subvec(0, kx - 1) = grad_beta;
        grad(kx) = grad_logtheta;

        // Return negative gradient for minimization
        grad = -grad;
    }
};

// Count model Geometric functor (special case of Negative Binomial with theta = 1)
class CountGeomFunctor : public LikelihoodFunctor
{
private:
    // Use a shared_ptr to manage the CountNegBinFunctor instance
    std::shared_ptr<CountNegBinFunctor> negbin_functor;

public:
    CountGeomFunctor(const arma::vec &y, const arma::mat &x, const arma::vec &offs, const arma::vec &w)
        : LikelihoodFunctor(y, x, offs, w),
          negbin_functor(std::make_shared<CountNegBinFunctor>(y, x, offs, w)) {}

    double operator()(const arma::vec &parms) override
    {
        // Create a new parameter vector with an additional element for theta = 1 (log(theta) = 0)
        arma::vec parms_extended(parms.n_elem + 1);
        parms_extended.subvec(0, parms.n_elem - 1) = parms;
        parms_extended(parms.n_elem) = 0.0; // log(1) = 0

        // Use the shared CountNegBinFunctor instance
        return (*negbin_functor)(parms_extended);
    }

    void Gradient(const arma::vec &parms, arma::vec &grad) override
    {
        // Create a new parameter vector with an additional element for theta = 1 (log(theta) = 0)
        arma::vec parms_extended(parms.n_elem + 1);
        parms_extended.subvec(0, parms.n_elem - 1) = parms;
        parms_extended(parms.n_elem) = 0.0; // log(1) = 0

        // Use the shared CountNegBinFunctor instance for gradient calculation
        arma::vec grad_extended;
        negbin_functor->Gradient(parms_extended, grad_extended);

        // Return only the gradient for the original parameters (exclude theta)
        grad = arma::zeros<arma::vec>(parms.n_elem);
        grad.subvec(0, parms.n_elem - 1) = grad_extended.subvec(0, parms.n_elem - 1);
    }
};

// Zero hurdle Poisson functor
class ZeroPoissonFunctor : public LikelihoodFunctor
{
public:
    ZeroPoissonFunctor(const arma::vec &y, const arma::mat &x, const arma::vec &offs, const arma::vec &w)
        : LikelihoodFunctor(y, x, offs, w) {}

    double operator()(const arma::vec &parms) override
    {
        // Calculate mu = exp(X * parms + offset)
        arma::vec eta = calculate_eta(parms);
        arma::vec mu = calculate_mu(eta);

        // Calculate log probability of zero: loglik0 = -mu
        arma::vec loglik0 = -mu;

        // Calculate log-likelihood
        double loglik = 0.0;

        // For Y=0 observations: sum(weights[Y0] * loglik0[Y0])
        if (Y0.n_elem > 0)
        {
            loglik += arma::dot(w_zero, loglik0.elem(Y0));
        }

        // For Y>0 observations: sum(weights[Y1] * log(1 - exp(loglik0[Y1])))
        if (Y1.n_elem > 0)
        {
            arma::vec temp = log1mexp(loglik0.elem(Y1));
            loglik += arma::dot(w_pos, temp);
        }

        // Return negative log-likelihood for minimization
        return -loglik;
    }

    void Gradient(const arma::vec &parms, arma::vec &grad) override
    {
        // Calculate eta = X * parms + offset
        arma::vec eta = calculate_eta(parms);

        // Calculate mu = exp(eta)
        arma::vec mu = calculate_mu(eta);

        // Initialize gradient term
        arma::vec grad_term = arma::zeros<arma::vec>(X.n_rows);

        // For Y=0 observations: -mu
        if (Y0.n_elem > 0)
        {
            grad_term.elem(Y0) = -mu.elem(Y0);
        }

        // For Y>0 observations: vectorized calculation
        if (Y1.n_elem > 0)
        {
            arma::vec mu_1 = mu.elem(Y1);
            arma::vec eta_1 = eta.elem(Y1);
            arma::vec loglik0 = -mu_1;
            arma::vec logratio = loglik0 - log1mexp(loglik0);
            grad_term.elem(Y1) = exp(logratio + eta_1);
        }

        // Calculate the gradient with single matrix multiplication
        grad = X.t() * (weights % grad_term);

        // Return negative gradient for minimization
        grad = -grad;
    }
};

// Zero hurdle Negative Binomial functor
class ZeroNegBinFunctor : public LikelihoodFunctor
{
public:
    ZeroNegBinFunctor(const arma::vec &y, const arma::mat &x, const arma::vec &offs, const arma::vec &w)
        : LikelihoodFunctor(y, x, offs, w) {}

    double operator()(const arma::vec &parms) override
    {
        // Get number of parameters
        int kz = X.n_cols;

        // Calculate mu = exp(X * parms[0:kz-1] + offset) for all observations
        arma::vec mu_zero, mu_pos;
        double theta = exp(parms(kz));
        double log_theta = log(theta);

        // Calculate log-likelihood
        double loglik = 0.0;

        // For Y=0 observations
        if (Y0.n_elem > 0)
        {
            mu_zero = calculate_mu(X_zero * parms.subvec(0, kz - 1) + offset_zero);
            arma::vec loglik0_zero = theta * log_theta - theta * log(theta + mu_zero);
            loglik += arma::dot(w_zero, loglik0_zero);
        }

        // For Y>0 observations
        if (Y1.n_elem > 0)
        {
            mu_pos = calculate_mu(X_pos * parms.subvec(0, kz - 1) + offset_pos);
            arma::vec loglik0_pos = theta * log_theta - theta * log(theta + mu_pos);
            arma::vec temp = log1mexp(loglik0_pos);
            loglik += arma::dot(w_pos, temp);
        }

        // Return negative log-likelihood for minimization
        return -loglik;
    }

    void Gradient(const arma::vec &parms, arma::vec &grad) override
    {
        // Get number of parameters
        int kz = X.n_cols;

        double theta = exp(parms(kz));
        double log_theta = log(theta);

        // Initialize gradient for beta parameters
        arma::vec grad_beta = arma::zeros<arma::vec>(kz);
        double grad_logtheta = 0.0;

        // For Y=0 observations
        if (Y0.n_elem > 0)
        {
            arma::vec eta_0 = X_zero * parms.subvec(0, kz - 1) + offset_zero;
            arma::vec mu_0 = calculate_mu(eta_0);

            arma::vec term1 = -mu_0 * theta / (mu_0 + theta);
            grad_beta += X_zero.t() * (w_zero % term1);

            arma::vec mu_theta_0 = mu_0 + theta;
            arma::vec term_theta_0 = log_theta - log(mu_theta_0) + 1.0 - theta / mu_theta_0;
            grad_logtheta += arma::dot(w_zero, term_theta_0);
        }

        // For Y>0 observations
        if (Y1.n_elem > 0)
        {
            arma::vec eta_1 = X_pos * parms.subvec(0, kz - 1) + offset_pos;
            arma::vec mu_1 = calculate_mu(eta_1);

            arma::vec loglik0_1 = theta * log_theta - theta * log(theta + mu_1);
            arma::vec logratio = loglik0_1 - log1mexp(loglik0_1);
            arma::vec term2 = exp(logratio + log_theta - log(mu_1 + theta) + eta_1);
            grad_beta += X_pos.t() * (w_pos % term2);

            arma::vec mu_theta_1 = mu_1 + theta;
            arma::vec term_theta_1 = exp(logratio) % (log_theta - log(mu_theta_1) + 1.0 - theta / mu_theta_1);
            grad_logtheta -= arma::dot(w_pos, term_theta_1);
        }

        grad_logtheta *= theta;

        // Combine gradients
        grad = arma::zeros<arma::vec>(parms.n_elem);
        grad.subvec(0, kz - 1) = grad_beta;
        grad(kz) = grad_logtheta;

        // Return negative gradient for minimization
        grad = -grad;
    }
};

// Zero hurdle Geometric functor (special case of Negative Binomial with theta = 1)
class ZeroGeomFunctor : public LikelihoodFunctor
{
private:
    // Use a shared_ptr to manage the ZeroNegBinFunctor instance
    std::shared_ptr<ZeroNegBinFunctor> negbin_functor;

public:
    ZeroGeomFunctor(const arma::vec &y, const arma::mat &x, const arma::vec &offs, const arma::vec &w)
        : LikelihoodFunctor(y, x, offs, w),
          negbin_functor(std::make_shared<ZeroNegBinFunctor>(y, x, offs, w)) {}

    double operator()(const arma::vec &parms) override
    {
        // Create a new parameter vector with an additional element for theta = 1 (log(theta) = 0)
        arma::vec parms_extended(parms.n_elem + 1);
        parms_extended.subvec(0, parms.n_elem - 1) = parms;
        parms_extended(parms.n_elem) = 0.0; // log(1) = 0

        // Use the shared ZeroNegBinFunctor instance
        return (*negbin_functor)(parms_extended);
    }

    void Gradient(const arma::vec &parms, arma::vec &grad) override
    {
        // Create a new parameter vector with an additional element for theta = 1 (log(theta) = 0)
        arma::vec parms_extended(parms.n_elem + 1);
        parms_extended.subvec(0, parms.n_elem - 1) = parms;
        parms_extended(parms.n_elem) = 0.0; // log(1) = 0

        // Use the shared ZeroNegBinFunctor instance for gradient calculation
        arma::vec grad_extended;
        negbin_functor->Gradient(parms_extended, grad_extended);

        // Return only the gradient for the original parameters (exclude theta)
        grad = arma::zeros<arma::vec>(parms.n_elem);
        grad.subvec(0, parms.n_elem - 1) = grad_extended.subvec(0, parms.n_elem - 1);
    }
};

// Zero hurdle Binomial functor
class ZeroBinomFunctor : public LikelihoodFunctor
{
private:
    links::LinkFunction linkinv_func;
    links::LinkFunction mu_eta_func;
    std::string link_name;
    bool is_logit; // Fast path for logit link

    // Cached binary indicator for logit fast path: 1.0 for Y>0, 0.0 for Y=0
    arma::vec Y_binary;

public:
    ZeroBinomFunctor(const arma::vec &y, const arma::mat &x, const arma::vec &offs, const arma::vec &w,
                     const std::string &link = "logit")
        : LikelihoodFunctor(y, x, offs, w), link_name(link), is_logit(link == "logit")
    {
        linkinv_func = links::get_linkinv(link);
        mu_eta_func = links::get_mu_eta(link);

        // For logit fast path, pre-compute binary indicator
        if (is_logit)
        {
            Y_binary = arma::conv_to<arma::vec>::from(Y > 0);
        }
    }

    double operator()(const arma::vec &parms) override
    {
        // Calculate eta = X * parms + offset
        arma::vec eta = calculate_eta(parms);

        if (is_logit)
        {
            // Logit fast path: use numerically stable softplus
            // log(sigmoid(x))  = -softplus(-x) where softplus(x) = log(1+exp(x))
            // log(1-sigmoid(x)) = -softplus(x)
            // Stabilized: softplus(x) = max(x,0) + log1p(exp(-|x|))
            double loglik = 0.0;
            if (Y1.n_elem > 0)
            {
                // -log(sigmoid(eta)) = softplus(-eta) = max(-eta,0) + log1p(exp(-|-eta|))
                arma::vec eta_pos = eta.elem(Y1);
                arma::vec sp = arma::max(-eta_pos, arma::zeros(eta_pos.n_elem)) +
                               log1p(exp(-abs(eta_pos)));
                loglik -= arma::dot(w_pos, sp);
            }
            if (Y0.n_elem > 0)
            {
                // -log(1-sigmoid(eta)) = softplus(eta) = max(eta,0) + log1p(exp(-|eta|))
                arma::vec eta_zero = eta.elem(Y0);
                arma::vec sp = arma::max(eta_zero, arma::zeros(eta_zero.n_elem)) +
                               log1p(exp(-abs(eta_zero)));
                loglik -= arma::dot(w_zero, sp);
            }
            return -loglik;
        }

        // General link path
        arma::vec mu = linkinv_func(eta);

        double loglik = 0.0;
        if (Y0.n_elem > 0)
        {
            arma::vec temp = log(1.0 - mu.elem(Y0));
            loglik += arma::dot(w_zero, temp);
        }
        if (Y1.n_elem > 0)
        {
            arma::vec temp = log(mu.elem(Y1));
            loglik += arma::dot(w_pos, temp);
        }
        return -loglik;
    }

    void Gradient(const arma::vec &parms, arma::vec &grad) override
    {
        // Calculate eta = X * parms + offset
        arma::vec eta = calculate_eta(parms);

        if (is_logit)
        {
            // Logit fast path: gradient = X' * (weights % (Y_binary - mu))
            // For logit: d/d(beta) = sum_i w_i * (y_i - mu_i) * x_i
            // where y_i is 1 for Y>0 and 0 for Y=0
            arma::vec mu = 1.0 / (1.0 + exp(-eta));
            grad = X.t() * (weights % (Y_binary - mu));
            grad = -grad;
            return;
        }

        // General link path
        arma::vec mu = arma::clamp(linkinv_func(eta), 1e-15, 1.0 - 1e-15);
        arma::vec mu_eta_vec = mu_eta_func(eta);

        arma::vec grad_term = arma::zeros<arma::vec>(X.n_rows);
        if (Y0.n_elem > 0)
        {
            grad_term.elem(Y0) = -1.0 / (1.0 - mu.elem(Y0));
        }
        if (Y1.n_elem > 0)
        {
            grad_term.elem(Y1) = 1.0 / mu.elem(Y1);
        }
        grad_term = grad_term % mu_eta_vec;

        grad = X.t() * (weights % grad_term);
        grad = -grad;
    }

    // Getter for link name
    std::string get_link_name() const
    {
        return link_name;
    }
};

// Combined functor for joint optimization
class JointFunctor : public Functor
{
private:
    std::shared_ptr<LikelihoodFunctor> count_functor;
    std::shared_ptr<LikelihoodFunctor> zero_functor;
    int kx;
    bool dist_negbin;

public:
    JointFunctor(std::shared_ptr<LikelihoodFunctor> count, std::shared_ptr<LikelihoodFunctor> zero,
                 int count_params, int zero_params,
                 bool count_is_negbin, bool zero_is_negbin)
        : count_functor(count), zero_functor(zero),
          kx(count_params),
          dist_negbin(count_is_negbin) {
        // Suppress unused parameter warnings
        (void)zero_params;
        (void)zero_is_negbin;
    }

    double operator()(const arma::vec &parms) override
    {
        // Split parameters for count and zero components
        arma::vec count_parms = parms.subvec(0, kx + (dist_negbin ? 0 : -1));
        arma::vec zero_parms = parms.subvec(kx + (dist_negbin ? 1 : 0), parms.n_elem - 1);

        // Calculate log-likelihood for both components
        double count_loglik = (*count_functor)(count_parms);
        double zero_loglik = (*zero_functor)(zero_parms);

        // Return combined negative log-likelihood
        return count_loglik + zero_loglik;
    }

    void Gradient(const arma::vec &parms, arma::vec &grad) override
    {
        // Split parameters for count and zero components
        arma::vec count_parms = parms.subvec(0, kx + (dist_negbin ? 0 : -1));
        arma::vec zero_parms = parms.subvec(kx + (dist_negbin ? 1 : 0), parms.n_elem - 1);

        // Calculate gradients for both components
        arma::vec count_grad, zero_grad;
        count_functor->Gradient(count_parms, count_grad);
        zero_functor->Gradient(zero_parms, zero_grad);

        // Combine gradients
        grad = arma::zeros<arma::vec>(parms.n_elem);
        grad.subvec(0, kx + (dist_negbin ? 0 : -1)) = count_grad;
        grad.subvec(kx + (dist_negbin ? 1 : 0), parms.n_elem - 1) = zero_grad;
    }
};

// Helper function to run optimization with any functor type
template <typename FunctorType>
Rcpp::List run_optimization(
    FunctorType &functor,
    arma::vec &start,
    const std::string &method = "BFGS",
    bool hessian = true,
    int maxit = 10000,
    double reltol = -1.0)
{
    // Create optimizer
    Roptim<FunctorType> opt(method);
    opt.control.trace = 0;
    opt.control.maxit = maxit;
    if (reltol > 0.0)
    {
        opt.control.reltol = reltol;
    }
    opt.set_hessian(hessian);

    // Optimize
    opt.minimize(functor, start);

    // Return results
    return Rcpp::List::create(
        Rcpp::Named("par") = opt.par(),
        Rcpp::Named("value") = -opt.value(), // Convert back to log-likelihood
        Rcpp::Named("counts") = Rcpp::List::create(
            Rcpp::Named("function") = opt.fncount(),
            Rcpp::Named("gradient") = opt.grcount()),
        Rcpp::Named("convergence") = opt.convergence(),
        Rcpp::Named("message") = opt.message(),
        Rcpp::Named("hessian") = opt.hessian());
}

// R interface functions for optimization using roptim

// [[Rcpp::export]]
Rcpp::List optim_count_poisson_cpp(
    const arma::vec &start,
    const arma::vec &Y,
    const arma::mat &X,
    const arma::vec &offsetx,
    const arma::vec &weights,
    const std::string &method = "BFGS",
    bool hessian = true,
    int maxit = 10000,
    double reltol = -1.0)
{
    // Create functor
    CountPoissonFunctor functor(Y, X, offsetx, weights);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List optim_count_negbin_cpp(
    const arma::vec &start,
    const arma::vec &Y,
    const arma::mat &X,
    const arma::vec &offsetx,
    const arma::vec &weights,
    const std::string &method = "BFGS",
    bool hessian = true,
    int maxit = 10000,
    double reltol = -1.0)
{
    // Create functor
    CountNegBinFunctor functor(Y, X, offsetx, weights);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List optim_count_geom_cpp(
    const arma::vec &start,
    const arma::vec &Y,
    const arma::mat &X,
    const arma::vec &offsetx,
    const arma::vec &weights,
    const std::string &method = "BFGS",
    bool hessian = true,
    int maxit = 10000,
    double reltol = -1.0)
{
    // Create functor
    CountGeomFunctor functor(Y, X, offsetx, weights);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List optim_zero_poisson_cpp(
    const arma::vec &start,
    const arma::vec &Y,
    const arma::mat &X,
    const arma::vec &offsetx,
    const arma::vec &weights,
    const std::string &method = "BFGS",
    bool hessian = true,
    int maxit = 10000,
    double reltol = -1.0)
{
    // Create functor
    ZeroPoissonFunctor functor(Y, X, offsetx, weights);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List optim_zero_negbin_cpp(
    const arma::vec &start,
    const arma::vec &Y,
    const arma::mat &X,
    const arma::vec &offsetx,
    const arma::vec &weights,
    const std::string &method = "BFGS",
    bool hessian = true,
    int maxit = 10000,
    double reltol = -1.0)
{
    // Create functor
    ZeroNegBinFunctor functor(Y, X, offsetx, weights);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List optim_zero_geom_cpp(
    const arma::vec &start,
    const arma::vec &Y,
    const arma::mat &X,
    const arma::vec &offsetx,
    const arma::vec &weights,
    const std::string &method = "BFGS",
    bool hessian = true,
    int maxit = 10000,
    double reltol = -1.0)
{
    // Create functor
    ZeroGeomFunctor functor(Y, X, offsetx, weights);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List optim_zero_binom_cpp(
    const arma::vec &start,
    const arma::vec &Y,
    const arma::mat &X,
    const arma::vec &offsetx,
    const arma::vec &weights,
    const std::string &link = "logit",
    const std::string &method = "BFGS",
    bool hessian = true,
    int maxit = 10000,
    double reltol = -1.0)
{
    // Create functor with C++ link function
    ZeroBinomFunctor functor(Y, X, offsetx, weights, link);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List optim_joint_cpp(
    const arma::vec &start,
    const arma::vec &Y,
    const arma::mat &X,
    const arma::vec &offsetx,
    const arma::mat &Z,
    const arma::vec &offsetz,
    const arma::vec &weights,
    const std::string &dist = "poisson",
    const std::string &zero_dist = "binomial",
    const std::string &link = "logit",
    const std::string &method = "BFGS",
    bool hessian = true,
    int maxit = 10000,
    double reltol = -1.0)
{
    // Create count functor based on distribution
    std::shared_ptr<LikelihoodFunctor> count_functor;
    bool dist_negbin = false;

    if (dist == "poisson")
    {
        count_functor = std::make_shared<CountPoissonFunctor>(Y, X, offsetx, weights);
    }
    else if (dist == "negbin")
    {
        count_functor = std::make_shared<CountNegBinFunctor>(Y, X, offsetx, weights);
        dist_negbin = true;
    }
    else if (dist == "geometric")
    {
        count_functor = std::make_shared<CountGeomFunctor>(Y, X, offsetx, weights);
    }
    else
    {
        Rcpp::stop("Unknown count distribution");
    }

    // Create zero functor based on distribution
    std::shared_ptr<LikelihoodFunctor> zero_functor;
    bool zero_dist_negbin = false;

    if (zero_dist == "poisson")
    {
        zero_functor = std::make_shared<ZeroPoissonFunctor>(Y, Z, offsetz, weights);
    }
    else if (zero_dist == "negbin")
    {
        zero_functor = std::make_shared<ZeroNegBinFunctor>(Y, Z, offsetz, weights);
        zero_dist_negbin = true;
    }
    else if (zero_dist == "geometric")
    {
        zero_functor = std::make_shared<ZeroGeomFunctor>(Y, Z, offsetz, weights);
    }
    else if (zero_dist == "binomial")
    {
        zero_functor = std::make_shared<ZeroBinomFunctor>(Y, Z, offsetz, weights, link);
    }
    else
    {
        Rcpp::stop("Unknown zero hurdle distribution");
    }

    // Create joint functor
    JointFunctor functor(count_functor, zero_functor, X.n_cols, Z.n_cols, dist_negbin, zero_dist_negbin);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List compute_negbin_hurdle_fitted_cpp(
    const arma::vec &coefc,
    const arma::vec &coefz,
    const arma::mat &X,
    const arma::mat &Z,
    const arma::vec &offsetx,
    const arma::vec &offsetz,
    double theta,
    const arma::vec &y)
{
    // Count component: mu = exp(X * coefc + offsetx)
    arma::vec mu = exp(X * coefc + offsetx);

    // Zero component (logit link): phi = sigmoid(Z * coefz + offsetz)
    arma::vec eta_z = Z * coefz + offsetz;
    arma::vec phi = 1.0 / (1.0 + exp(-eta_z));

    // log P(Y > 0 | NB) = log(1 - (theta/(theta+mu))^theta)
    // = log1mexp(theta * log(theta/(theta+mu)))
    // = log1mexp(theta * (log(theta) - log(theta+mu)))
    arma::vec log_p0_nb = theta * (log(theta) - log(theta + mu));
    arma::vec log_p1_nb = log1mexp(log_p0_nb);

    // Fitted values: Yhat = exp(log(phi) - log_p1_nb + log(mu))
    arma::vec Yhat = exp(log(phi) - log_p1_nb + log(mu));

    // Residuals
    arma::vec res = y - Yhat;

    return Rcpp::List::create(
        Rcpp::Named("fitted.values") = Yhat,
        Rcpp::Named("residuals") = res);
}
