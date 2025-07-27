#include <RcppArmadillo.h>
#include <roptim.h>
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

// Helper function for negative binomial log probability
inline double dnbinom_log(double y, double size, double mu)
{
    return lgamma(y + size) - lgamma(size) - lgamma(y + 1) +
           size * log(size) + y * log(mu) - (y + size) * log(size + mu);
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

public:
    LikelihoodFunctor(const arma::vec &y, const arma::mat &x, const arma::vec &offs, const arma::vec &w)
        : Y(y), X(x), offset(offs), weights(w)
    {
        // Pre-compute indicator vectors
        Y0 = find(Y <= 0);
        Y1 = find(Y > 0);
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
public:
    CountPoissonFunctor(const arma::vec &y, const arma::mat &x, const arma::vec &offs, const arma::vec &w)
        : LikelihoodFunctor(y, x, offs, w) {}

    double operator()(const arma::vec &parms) override
    {
        // If no Y>0 observations, return 0
        if (Y1.n_elem == 0)
        {
            return 0.0;
        }

        // Calculate mu = exp(X * parms + offset) for Y>0 observations
        arma::vec mu = calculate_mu(X.rows(Y1) * parms + offset.elem(Y1));

        // Calculate log probability of zero: loglik0 = -mu
        arma::vec loglik0 = -mu;

        // Get Y values for positive observations
        arma::vec Y_pos = Y.elem(Y1);

        // Vectorized log probability of Y[Y1]: dpois(y, lambda, log) = y * log(lambda) - lambda - lgamma(y + 1)
        // First compute lgamma for Y_pos + 1
        arma::vec lgamma_y_1(Y1.n_elem);
        for (size_t i = 0; i < Y1.n_elem; i++)
        {
            lgamma_y_1(i) = lgamma(Y_pos(i) + 1.0);
        }

        arma::vec loglik1 = Y_pos % log(mu) - mu - lgamma_y_1;

        // Calculate log-likelihood
        double loglik = sum(weights.elem(Y1) % loglik1) - sum(weights.elem(Y1) % log(1.0 - exp(loglik0)));

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
        arma::vec eta = X.rows(Y1) * parms + offset.elem(Y1);

        // Calculate mu = exp(eta)
        arma::vec mu = calculate_mu(eta);

        // Get Y values for positive observations
        arma::vec Y_pos = Y.elem(Y1);

        // Vectorized gradient calculation
        arma::vec loglik0 = -mu; // log probability of zero
        arma::vec grad_term = Y_pos - mu - exp(loglik0 - log(1.0 - exp(loglik0)) + eta);

        // Single matrix multiplication instead of loop
        grad = X.rows(Y1).t() * (weights.elem(Y1) % grad_term);

        // Return negative gradient for minimization
        grad = -grad;
    }
};

// Count model Negative Binomial functor
class CountNegBinFunctor : public LikelihoodFunctor
{
public:
    CountNegBinFunctor(const arma::vec &y, const arma::mat &x, const arma::vec &offs, const arma::vec &w)
        : LikelihoodFunctor(y, x, offs, w) {}

    double operator()(const arma::vec &parms) override
    {
        // Get number of parameters
        int kx = X.n_cols;

        // If no Y>0 observations, return 0
        if (Y1.n_elem == 0)
        {
            return 0.0;
        }

        // Calculate mu = exp(X * parms[0:kx-1] + offset) for Y>0 observations
        arma::vec mu = calculate_mu(X.rows(Y1) * parms.subvec(0, kx - 1) + offset.elem(Y1));

        // Calculate theta = exp(parms[kx])
        double theta = exp(parms(kx));

        // Get Y values for positive observations
        arma::vec Y_pos = Y.elem(Y1);

        // Vectorized log probability of zero
        arma::vec loglik0 = theta * log(theta) - theta * log(theta + mu);

        // Vectorized log probability of Y[Y1]
        // First compute lgamma for vectors
        arma::vec lgamma_y_theta(Y1.n_elem);
        arma::vec lgamma_y_1(Y1.n_elem);
        for (size_t i = 0; i < Y1.n_elem; i++)
        {
            lgamma_y_theta(i) = lgamma(Y_pos(i) + theta);
            lgamma_y_1(i) = lgamma(Y_pos(i) + 1.0);
        }

        // Vectorized negative binomial log probability
        arma::vec loglik1 = lgamma_y_theta - lgamma(theta) - lgamma_y_1 +
                            Y_pos % log(mu) - (Y_pos + theta) % log(mu + theta);

        // Calculate log-likelihood
        double loglik = sum(weights.elem(Y1) % loglik1) - sum(weights.elem(Y1) % log(1.0 - exp(loglik0)));

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

        // Calculate eta = X * parms[0:kx-1] + offset for Y>0 observations
        arma::vec eta = X.rows(Y1) * parms.subvec(0, kx - 1) + offset.elem(Y1);

        // Calculate mu = exp(eta)
        arma::vec mu = calculate_mu(eta);

        // Calculate theta = exp(parms[kx])
        double theta = exp(parms(kx));

        // Vectorized calculation of log probability of zero
        arma::vec loglik0 = theta * log(theta) - theta * log(theta + mu);

        // Calculate logratio = log(p0/(1-p0))
        arma::vec logratio = loglik0 - log(1.0 - exp(loglik0));

        // Get Y values for positive observations
        arma::vec Y_pos = Y.elem(Y1);

        // Vectorized gradient calculation for beta parameters
        arma::vec grad_term = Y_pos - mu % (Y_pos + theta) / (mu + theta) -
                              exp(logratio + log(theta) - log(mu + theta) + eta);

        // Single matrix multiplication instead of loop
        arma::vec grad_beta = X.rows(Y1).t() * (weights.elem(Y1) % grad_term);

        // Vectorized gradient calculation for log(theta)
        arma::vec mu_plus_theta = mu + theta;
        arma::vec log_mu_plus_theta = log(mu_plus_theta);

        // Vectorized digamma for Y_pos + theta
        arma::vec digamma_y_theta(Y1.n_elem);
        for (size_t i = 0; i < Y1.n_elem; i++)
        {
            digamma_y_theta(i) = R::digamma(Y_pos(i) + theta);
        }

        double digamma_theta = R::digamma(theta);

        // Vectorized first term for grad_logtheta
        arma::vec term3 = digamma_y_theta - digamma_theta + log(theta) - log_mu_plus_theta +
                          1.0 - (Y_pos + theta) / mu_plus_theta;

        // Vectorized second term for grad_logtheta
        arma::vec term4 = exp(logratio) % (log(theta) - log_mu_plus_theta + 1.0 - theta / mu_plus_theta);

        // Sum with weights
        double grad_logtheta = theta * sum(weights.elem(Y1) % (term3 + term4));

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
public:
    CountGeomFunctor(const arma::vec &y, const arma::mat &x, const arma::vec &offs, const arma::vec &w)
        : LikelihoodFunctor(y, x, offs, w) {}

    double operator()(const arma::vec &parms) override
    {
        // Create a new parameter vector with an additional element for theta = 1 (log(theta) = 0)
        arma::vec parms_extended(parms.n_elem + 1);
        parms_extended.subvec(0, parms.n_elem - 1) = parms;
        parms_extended(parms.n_elem) = 0.0; // log(1) = 0

        // Create a CountNegBinFunctor and call it with the extended parameter vector
        CountNegBinFunctor functor(Y, X, offset, weights);
        return functor(parms_extended);
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
        arma::vec eta = X.rows(Y1) * parms + offset.elem(Y1);

        // Calculate mu = exp(eta)
        arma::vec mu = exp(eta);

        // Get Y values for positive observations
        arma::vec Y_pos = Y.elem(Y1);

        // Vectorized calculation of the first term: Y[Y1] - mu * (Y[Y1] + 1) / (mu + 1)
        arma::vec mu_plus_one = mu + 1.0;
        arma::vec term1 = Y_pos - mu % (Y_pos + 1.0) / mu_plus_one;

        // Vectorized calculation of log probability of zero for geometric (theta=1)
        arma::vec loglik0 = log(1.0) - log(mu_plus_one); // Simplified: log(1/(mu+1))

        // Vectorized second term
        arma::vec logratio = loglik0 - log(1.0 - exp(loglik0));
        arma::vec term2 = exp(logratio - log(mu_plus_one) + eta);

        // Vectorized gradient calculation
        arma::vec grad_term = term1 - term2;

        // Single matrix multiplication
        grad = X.rows(Y1).t() * (weights.elem(Y1) % grad_term);

        // Return negative gradient for minimization
        grad = -grad;
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
            loglik += sum(weights.elem(Y0) % loglik0.elem(Y0));
        }

        // For Y>0 observations: sum(weights[Y1] * log(1 - exp(loglik0[Y1])))
        if (Y1.n_elem > 0)
        {
            arma::vec temp = log(1.0 - exp(loglik0.elem(Y1)));
            loglik += sum(weights.elem(Y1) % temp);
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
            arma::vec logratio = loglik0 - log(1.0 - exp(loglik0));
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

        // Calculate mu = exp(X * parms[0:kz-1] + offset)
        arma::vec mu = calculate_mu(X * parms.subvec(0, kz - 1) + offset);

        // Calculate theta = exp(parms[kz])
        double theta = exp(parms(kz));

        // Calculate log-likelihood
        double loglik = 0.0;

        // Vectorized log probability of zero
        arma::vec loglik0 = theta * log(theta) - theta * log(theta + mu);

        // For Y=0 observations: sum(weights[Y0] * loglik0[Y0])
        if (Y0.n_elem > 0)
        {
            loglik += sum(weights.elem(Y0) % loglik0.elem(Y0));
        }

        // For Y>0 observations: sum(weights[Y1] * log(1 - exp(loglik0[Y1])))
        if (Y1.n_elem > 0)
        {
            arma::vec temp = log(1.0 - exp(loglik0.elem(Y1)));
            loglik += sum(weights.elem(Y1) % temp);
        }

        // Return negative log-likelihood for minimization
        return -loglik;
    }

    void Gradient(const arma::vec &parms, arma::vec &grad) override
    {
        // Get number of parameters
        int kz = X.n_cols;

        // Calculate eta = X * parms[0:kz-1] + offset
        arma::vec eta = X * parms.subvec(0, kz - 1) + offset;

        // Calculate mu = exp(eta)
        arma::vec mu = calculate_mu(eta);

        // Calculate theta = exp(parms[kz])
        double theta = exp(parms(kz));

        // Vectorized log probability of zero
        arma::vec loglik0 = theta * log(theta) - theta * log(theta + mu);

        // Initialize gradient for beta parameters
        arma::vec grad_beta = arma::zeros<arma::vec>(kz);

        // For Y=0 observations: -mu * theta / (mu + theta)
        if (Y0.n_elem > 0)
        {
            arma::vec mu_0 = mu.elem(Y0);
            arma::vec term1 = -mu_0 * theta / (mu_0 + theta);
            grad_beta += X.rows(Y0).t() * (weights.elem(Y0) % term1);
        }

        // For Y>0 observations
        if (Y1.n_elem > 0)
        {
            arma::vec mu_1 = mu.elem(Y1);
            arma::vec loglik0_1 = loglik0.elem(Y1);
            arma::vec logratio = loglik0_1 - log(1.0 - exp(loglik0_1));
            arma::vec term2 = exp(logratio + log(theta) - log(mu_1 + theta) + eta.elem(Y1));
            grad_beta += X.rows(Y1).t() * (weights.elem(Y1) % term2);
        }

        // Vectorized gradient for log(theta)
        double grad_logtheta = 0.0;

        // For Y=0 observations
        if (Y0.n_elem > 0)
        {
            arma::vec mu_0 = mu.elem(Y0);
            arma::vec mu_theta_0 = mu_0 + theta;
            arma::vec term_theta_0 = log(theta) - log(mu_theta_0) + 1.0 - theta / mu_theta_0;
            grad_logtheta += sum(weights.elem(Y0) % term_theta_0);
        }

        // For Y>0 observations
        if (Y1.n_elem > 0)
        {
            arma::vec mu_1 = mu.elem(Y1);
            arma::vec mu_theta_1 = mu_1 + theta;
            arma::vec loglik0_1 = loglik0.elem(Y1);
            arma::vec logratio = loglik0_1 - log(1.0 - exp(loglik0_1));
            arma::vec term_theta_1 = exp(logratio) % (log(theta) - log(mu_theta_1) + 1.0 - theta / mu_theta_1);
            grad_logtheta -= sum(weights.elem(Y1) % term_theta_1);
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

public:
    ZeroBinomFunctor(const arma::vec &y, const arma::mat &x, const arma::vec &offs, const arma::vec &w,
                     const std::string &link = "logit")
        : LikelihoodFunctor(y, x, offs, w), link_name(link)
    {
        linkinv_func = links::get_linkinv(link);
        mu_eta_func = links::get_mu_eta(link);
    }

    double operator()(const arma::vec &parms) override
    {
        // Calculate eta = X * parms + offset
        arma::vec eta = calculate_eta(parms);

        // Calculate mu = linkinv(eta)
        arma::vec mu = linkinv_func(eta);

        // Calculate log-likelihood
        double loglik = 0.0;

        // For Y=0 observations: sum(weights[Y0] * log(1 - mu[Y0]))
        if (Y0.n_elem > 0)
        {
            arma::vec temp = log(1.0 - mu.elem(Y0));
            loglik += sum(weights.elem(Y0) % temp);
        }

        // For Y>0 observations: sum(weights[Y1] * log(mu[Y1]))
        if (Y1.n_elem > 0)
        {
            arma::vec temp = log(mu.elem(Y1));
            loglik += sum(weights.elem(Y1) % temp);
        }

        // Return negative log-likelihood for minimization
        return -loglik;
    }

    void Gradient(const arma::vec &parms, arma::vec &grad) override
    {
        // Calculate eta = X * parms + offset
        arma::vec eta = calculate_eta(parms);

        // Calculate mu = linkinv(eta)
        arma::vec mu = linkinv_func(eta);

        // Calculate mu_eta (derivative of mu with respect to eta)
        arma::vec mu_eta_vec = mu_eta_func(eta);

        // Initialize gradient term
        arma::vec grad_term = arma::zeros<arma::vec>(X.n_rows);

        // For Y=0 observations: -1 / (1 - mu)
        if (Y0.n_elem > 0)
        {
            grad_term.elem(Y0) = -1.0 / (1.0 - mu.elem(Y0));
        }

        // For Y>0 observations: 1 / mu
        if (Y1.n_elem > 0)
        {
            grad_term.elem(Y1) = 1.0 / mu.elem(Y1);
        }

        // Apply mu_eta (derivative of mu with respect to eta)
        grad_term = grad_term % mu_eta_vec;

        // Calculate the gradient
        grad = X.t() * (weights % grad_term);

        // Return negative gradient for minimization
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
    int kz;
    bool dist_negbin;
    bool zero_dist_negbin;

public:
    JointFunctor(std::shared_ptr<LikelihoodFunctor> count, std::shared_ptr<LikelihoodFunctor> zero,
                 int count_params, int zero_params,
                 bool count_is_negbin, bool zero_is_negbin)
        : count_functor(count), zero_functor(zero),
          kx(count_params), kz(zero_params),
          dist_negbin(count_is_negbin), zero_dist_negbin(zero_is_negbin) {}

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
    bool hessian = true)
{
    // Create optimizer
    Roptim<FunctorType> opt(method);
    opt.control.trace = 0;
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
    bool hessian = true)
{
    // Create functor
    CountPoissonFunctor functor(Y, X, offsetx, weights);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian);
}

// [[Rcpp::export]]
Rcpp::List optim_count_negbin_cpp(
    const arma::vec &start,
    const arma::vec &Y,
    const arma::mat &X,
    const arma::vec &offsetx,
    const arma::vec &weights,
    const std::string &method = "BFGS",
    bool hessian = true)
{
    // Create functor
    CountNegBinFunctor functor(Y, X, offsetx, weights);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian);
}

// [[Rcpp::export]]
Rcpp::List optim_count_geom_cpp(
    const arma::vec &start,
    const arma::vec &Y,
    const arma::mat &X,
    const arma::vec &offsetx,
    const arma::vec &weights,
    const std::string &method = "BFGS",
    bool hessian = true)
{
    // Create functor
    CountGeomFunctor functor(Y, X, offsetx, weights);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian);
}

// [[Rcpp::export]]
Rcpp::List optim_zero_poisson_cpp(
    const arma::vec &start,
    const arma::vec &Y,
    const arma::mat &X,
    const arma::vec &offsetx,
    const arma::vec &weights,
    const std::string &method = "BFGS",
    bool hessian = true)
{
    // Create functor
    ZeroPoissonFunctor functor(Y, X, offsetx, weights);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian);
}

// [[Rcpp::export]]
Rcpp::List optim_zero_negbin_cpp(
    const arma::vec &start,
    const arma::vec &Y,
    const arma::mat &X,
    const arma::vec &offsetx,
    const arma::vec &weights,
    const std::string &method = "BFGS",
    bool hessian = true)
{
    // Create functor
    ZeroNegBinFunctor functor(Y, X, offsetx, weights);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian);
}

// [[Rcpp::export]]
Rcpp::List optim_zero_geom_cpp(
    const arma::vec &start,
    const arma::vec &Y,
    const arma::mat &X,
    const arma::vec &offsetx,
    const arma::vec &weights,
    const std::string &method = "BFGS",
    bool hessian = true)
{
    // Create functor
    ZeroGeomFunctor functor(Y, X, offsetx, weights);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian);
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
    bool hessian = true)
{
    // Create functor with C++ link function
    ZeroBinomFunctor functor(Y, X, offsetx, weights, link);

    // Run optimization
    arma::vec par = start;
    return run_optimization(functor, par, method, hessian);
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
    bool hessian = true)
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
    return run_optimization(functor, par, method, hessian);
}
