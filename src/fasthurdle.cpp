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
namespace links {
// Link function type definitions
using LinkFunction = std::function<arma::vec(const arma::vec &)>;

// Link function registry
struct LinkFunctions {
  LinkFunction linkinv;
  LinkFunction mu_eta;
};

// Logit link
inline arma::vec logit_linkinv(const arma::vec &eta) {
  return 1.0 / (1.0 + exp(-eta));
}

inline arma::vec logit_mu_eta(const arma::vec &eta) {
  arma::vec mu = logit_linkinv(eta);
  return mu % (1.0 - mu);
}

// Probit link
inline arma::vec probit_linkinv(const arma::vec &eta) {
  arma::vec result(eta.n_elem);
  for (size_t i = 0; i < eta.n_elem; i++) {
    result(i) = R::pnorm(eta(i), 0.0, 1.0, 1, 0);
  }
  return result;
}

inline arma::vec probit_mu_eta(const arma::vec &eta) {
  arma::vec result(eta.n_elem);
  for (size_t i = 0; i < eta.n_elem; i++) {
    result(i) = R::dnorm(eta(i), 0.0, 1.0, 0);
  }
  return result;
}

// Complementary log-log link
inline arma::vec cloglog_linkinv(const arma::vec &eta) {
  return 1.0 - exp(-exp(eta));
}

inline arma::vec cloglog_mu_eta(const arma::vec &eta) {
  return exp(eta - exp(eta));
}

// Cauchit link
inline arma::vec cauchit_linkinv(const arma::vec &eta) {
  arma::vec result(eta.n_elem);
  for (size_t i = 0; i < eta.n_elem; i++) {
    result(i) = R::pcauchy(eta(i), 0.0, 1.0, 1, 0);
  }
  return result;
}

inline arma::vec cauchit_mu_eta(const arma::vec &eta) {
  arma::vec result(eta.n_elem);
  for (size_t i = 0; i < eta.n_elem; i++) {
    result(i) = R::dcauchy(eta(i), 0.0, 1.0, 0);
  }
  return result;
}

// Log link
inline arma::vec log_linkinv(const arma::vec &eta) { return exp(eta); }

inline arma::vec log_mu_eta(const arma::vec &eta) { return exp(eta); }

// Link function registry
static const std::unordered_map<std::string, LinkFunctions> link_registry = {
    {"logit", {logit_linkinv, logit_mu_eta}},
    {"probit", {probit_linkinv, probit_mu_eta}},
    {"cloglog", {cloglog_linkinv, cloglog_mu_eta}},
    {"cauchit", {cauchit_linkinv, cauchit_mu_eta}},
    {"log", {log_linkinv, log_mu_eta}}};

// Get link function by name
inline LinkFunction get_linkinv(const std::string &link) {
  auto it = link_registry.find(link);
  if (it != link_registry.end()) {
    return it->second.linkinv;
  }
  Rcpp::stop("Unknown link function: " + link);
  return nullptr;
}

inline LinkFunction get_mu_eta(const std::string &link) {
  auto it = link_registry.find(link);
  if (it != link_registry.end()) {
    return it->second.mu_eta;
  }
  Rcpp::stop("Unknown link function: " + link);
  return nullptr;
}
}  // namespace links

// Numerically stable log(1 - exp(x)) for x < 0
// Uses log(-expm1(x)) when x is close to 0 (x > -ln2)
// and log1p(-exp(x)) when x is very negative (x <= -ln2)
inline arma::vec log1mexp(const arma::vec &x) {
  arma::vec result(x.n_elem);
  // Split into two branches for vectorization
  arma::uvec small = find(x > -M_LN2);   // close to 0: use expm1
  arma::uvec large = find(x <= -M_LN2);  // very negative: use exp

  if (small.n_elem > 0) {
    // For each element in 'small', compute log(-expm1(x))
    arma::vec xs = x.elem(small);
    for (size_t i = 0; i < xs.n_elem; i++) {
      xs(i) = std::log(-std::expm1(xs(i)));
    }
    result.elem(small) = xs;
  }
  if (large.n_elem > 0) {
    // Vectorized path: log1p(-exp(x)) for the common case
    result.elem(large) = log1p(-exp(x.elem(large)));
  }
  return result;
}

// Base class for all likelihood functors
class LikelihoodFunctor : public Functor {
 protected:
  const arma::vec &Y;
  const arma::mat &X;
  const arma::vec &offset;
  const arma::vec &weights;

  // Cached indicator vectors for Y=0 and Y>0
  arma::uvec Y0;
  arma::uvec Y1;

  // Cached subsets (constant across all iterations)
  arma::mat X_pos;        // X.rows(Y1)
  arma::mat X_zero;       // X.rows(Y0)
  arma::vec offset_pos;   // offset.elem(Y1)
  arma::vec offset_zero;  // offset.elem(Y0)
  arma::vec w_pos;        // weights.elem(Y1)
  arma::vec w_zero;       // weights.elem(Y0)
  arma::vec Y_pos;        // Y.elem(Y1)

 public:
  LikelihoodFunctor(const arma::vec &y, const arma::mat &x,
                    const arma::vec &offs, const arma::vec &w)
      : Y(y), X(x), offset(offs), weights(w) {
    // Pre-compute indicator vectors
    Y0 = find(Y <= 0);
    Y1 = find(Y > 0);

    // Cache constant subsets to avoid repeated allocation
    if (Y1.n_elem > 0) {
      X_pos = X.rows(Y1);
      offset_pos = offset.elem(Y1);
      w_pos = weights.elem(Y1);
      Y_pos = Y.elem(Y1);
    }
    if (Y0.n_elem > 0) {
      X_zero = X.rows(Y0);
      offset_zero = offset.elem(Y0);
      w_zero = weights.elem(Y0);
    }
  }

  // Common utility functions
  arma::vec calculate_eta(const arma::vec &parms) const {
    return X * parms + offset;
  }

  arma::vec calculate_mu(const arma::vec &eta) const { return exp(eta); }

  // Virtual destructor for proper cleanup in derived classes
  virtual ~LikelihoodFunctor() = default;
};

// Count model Poisson functor
class CountPoissonFunctor : public LikelihoodFunctor {
 private:
  arma::vec lgamma_y_1_cached;  // lgamma(Y_pos + 1), constant across iterations

 public:
  CountPoissonFunctor(const arma::vec &y, const arma::mat &x,
                      const arma::vec &offs, const arma::vec &w)
      : LikelihoodFunctor(y, x, offs, w) {
    if (Y1.n_elem > 0) {
      lgamma_y_1_cached.set_size(Y1.n_elem);
      for (size_t i = 0; i < Y1.n_elem; i++) {
        lgamma_y_1_cached(i) = lgamma(Y_pos(i) + 1.0);
      }
    }
  }

  double operator()(const arma::vec &parms) override {
    // If no Y>0 observations, return 0
    if (Y1.n_elem == 0) {
      return 0.0;
    }

    // Calculate mu = exp(X * parms + offset) for Y>0 observations
    arma::vec mu = calculate_mu(X_pos * parms + offset_pos);

    // Calculate log probability of zero: loglik0 = -mu
    arma::vec loglik0 = -mu;

    arma::vec loglik1 = Y_pos % log(mu) - mu - lgamma_y_1_cached;

    // Calculate log-likelihood
    double loglik =
        arma::dot(w_pos, loglik1) - arma::dot(w_pos, log1mexp(loglik0));

    // Return negative log-likelihood for minimization
    return -loglik;
  }

  void Gradient(const arma::vec &parms, arma::vec &grad) override {
    // If no Y>0 observations, return zero gradient
    if (Y1.n_elem == 0) {
      grad = arma::zeros<arma::vec>(parms.n_elem);
      return;
    }

    // Calculate eta = X * parms + offset for Y>0 observations
    arma::vec eta = X_pos * parms + offset_pos;

    // Calculate mu = exp(eta)
    arma::vec mu = calculate_mu(eta);

    // Vectorized gradient calculation
    arma::vec loglik0 = -mu;  // log probability of zero
    arma::vec grad_term = Y_pos - mu - exp(loglik0 - log1mexp(loglik0) + eta);

    // Single matrix multiplication instead of loop
    grad = X_pos.t() * (w_pos % grad_term);

    // Return negative gradient for minimization
    grad = -grad;
  }
};

// Count model Negative Binomial functor
class CountNegBinFunctor : public LikelihoodFunctor {
 private:
  arma::vec lgamma_y_1_cached;  // lgamma(Y_pos + 1), constant across iterations

  // Unique-value lookup for Y_pos: compute expensive special functions
  // (lgamma, digamma) only for distinct Y values, then scatter back.
  arma::vec unique_y_vals;  // sorted unique values of Y_pos
  arma::uvec
      y_index_map;  // maps each Y_pos element to its index in unique_y_vals

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

  void compute_intermediates(const arma::vec &parms) {
    int kx = X.n_cols;
    cached_eta = X_pos * parms.subvec(0, kx - 1) + offset_pos;
    cached_mu = exp(cached_eta);
    cached_theta = exp(parms(kx));
    double log_theta = log(cached_theta);
    cached_loglik0 =
        cached_theta * log_theta - cached_theta * log(cached_theta + cached_mu);
    cached_logratio = cached_loglik0 - log1mexp(cached_loglik0);
    cached_parms = parms;
    cache_valid = true;
  }

 public:
  CountNegBinFunctor(const arma::vec &y, const arma::mat &x,
                     const arma::vec &offs, const arma::vec &w)
      : LikelihoodFunctor(y, x, offs, w),
        cached_theta(0.0),
        cache_valid(false) {
    if (Y1.n_elem > 0) {
      // Build unique-value lookup for Y_pos
      unique_y_vals = arma::unique(Y_pos);  // returns sorted
      y_index_map.set_size(Y_pos.n_elem);
      for (size_t i = 0; i < Y_pos.n_elem; i++) {
        // Binary search since unique_y_vals is sorted
        auto it = std::lower_bound(unique_y_vals.begin(), unique_y_vals.end(),
                                   Y_pos(i));
        y_index_map(i) = static_cast<arma::uword>(it - unique_y_vals.begin());
      }

      // Cache lgamma(Y_pos + 1) using unique-value lookup
      arma::vec lgamma_unique(unique_y_vals.n_elem);
      for (size_t j = 0; j < unique_y_vals.n_elem; j++) {
        lgamma_unique(j) = lgamma(unique_y_vals(j) + 1.0);
      }
      lgamma_y_1_cached = lgamma_unique.elem(y_index_map);
    }
  }

  double operator()(const arma::vec &parms) override {
    // If no Y>0 observations, return 0
    if (Y1.n_elem == 0) {
      return 0.0;
    }

    // Compute and cache intermediates
    compute_intermediates(parms);

    // Compute lgamma(Y + theta) only for unique Y values, then scatter
    arma::vec lgamma_unique(unique_y_vals.n_elem);
    for (size_t j = 0; j < unique_y_vals.n_elem; j++) {
      lgamma_unique(j) = lgamma(unique_y_vals(j) + cached_theta);
    }
    arma::vec lgamma_y_theta = lgamma_unique.elem(y_index_map);

    // Vectorized negative binomial log probability
    arma::vec loglik1 = lgamma_y_theta - lgamma(cached_theta) -
                        lgamma_y_1_cached + Y_pos % log(cached_mu) +
                        cached_theta * log(cached_theta) -
                        (Y_pos + cached_theta) % log(cached_mu + cached_theta);

    // Calculate log-likelihood
    double loglik =
        arma::dot(w_pos, loglik1) - arma::dot(w_pos, log1mexp(cached_loglik0));

    // Return negative log-likelihood for minimization
    return -loglik;
  }

  void Gradient(const arma::vec &parms, arma::vec &grad) override {
    // Get number of parameters
    int kx = X.n_cols;

    // If no Y>0 observations, return zero gradient
    if (Y1.n_elem == 0) {
      grad = arma::zeros<arma::vec>(parms.n_elem);
      return;
    }

    // Reuse cached intermediates from operator() if parameters match
    if (!cache_valid || parms.n_elem != cached_parms.n_elem ||
        !arma::all(parms == cached_parms)) {
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
    for (size_t j = 0; j < unique_y_vals.n_elem; j++) {
      digamma_unique(j) = R::digamma(unique_y_vals(j) + theta);
    }
    arma::vec digamma_y_theta = digamma_unique.elem(y_index_map);

    double digamma_theta = R::digamma(theta);

    // Vectorized first term for grad_logtheta
    arma::vec term3 = digamma_y_theta - digamma_theta + log(theta) -
                      log_mu_plus_theta + 1.0 - (Y_pos + theta) / mu_plus_theta;

    // Vectorized second term for grad_logtheta
    arma::vec term4 = exp(logratio) % (log(theta) - log_mu_plus_theta + 1.0 -
                                       theta / mu_plus_theta);

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

// ==========================================================================
// Expected Fisher Information for zero-truncated NB model
// ==========================================================================

// Per-observation FIM components for zero-truncated NB.
struct ZtnbFimResult {
  arma::vec v_ee;   // E[s_eta^2] per observation (weighted)
  arma::vec v_et;   // E[s_eta * s_logtheta] per observation (weighted)
  double v_tt_sum;  // sum_i w_i * E[s_logtheta^2]
};

// Compute expected Fisher information components for zero-truncated NB model.
// Uses PMF recurrence (no lgamma in inner loop). Cost: O(n_pos * avg_y_max).
ZtnbFimResult compute_ztnb_fim_components(const arma::vec &beta, double theta,
                                          const arma::mat &X_pos,
                                          const arma::vec &offset_pos,
                                          const arma::vec &w_pos,
                                          double quantile_cutoff = 0.9999) {
  int n_pos = X_pos.n_rows;
  double log_theta = std::log(theta);
  double digamma_theta = R::digamma(theta);

  ZtnbFimResult result;
  result.v_ee.zeros(n_pos);
  result.v_et.zeros(n_pos);
  result.v_tt_sum = 0.0;

  for (int i = 0; i < n_pos; i++) {
    double eta_i = arma::dot(X_pos.row(i), beta) + offset_pos(i);
    double mu = std::exp(eta_i);
    double mu_theta = mu + theta;
    double mu_over_mutheta = mu / mu_theta;
    double theta_over_mutheta = theta / mu_theta;

    double log_p0 = theta * (log_theta - std::log(mu_theta));
    double p0 = std::exp(log_p0);
    double p1 = 1.0 - p0;
    if (p1 < 1e-300) p1 = 1e-300;
    double r = p0 / p1;

    int y_max = static_cast<int>(
        R::qnbinom(quantile_cutoff, theta, theta / mu_theta, 1, 0));
    if (y_max < 1) y_max = 1;
    if (y_max > 10000) y_max = 10000;

    double c_trunc_eta = r * theta * mu_over_mutheta;
    double log_ratio = log_theta - std::log(mu_theta);
    double c_trunc_logtheta = r * (log_ratio + 1.0 - theta_over_mutheta);

    double pmf = std::exp(R::dnbinom_mu(1.0, theta, mu, 1) - std::log(p1));
    double digamma_y_theta = R::digamma(1.0 + theta);
    double pmf_ratio = mu_over_mutheta;

    double ee_i = 0.0, et_i = 0.0, tt_i = 0.0;

    for (int y = 1; y <= y_max; y++) {
      if (y > 1) {
        pmf *= (y - 1.0 + theta) / static_cast<double>(y) * pmf_ratio;
        digamma_y_theta += 1.0 / (y - 1.0 + theta);
      }
      if (pmf < 1e-300) continue;

      double y_d = static_cast<double>(y);
      double s_eta = y_d - mu * (y_d + theta) / mu_theta - c_trunc_eta;
      double s_logtheta =
          theta * (digamma_y_theta - digamma_theta + log_ratio + 1.0 -
                   (y_d + theta) / mu_theta + c_trunc_logtheta);

      double pmf_seta = pmf * s_eta;
      ee_i += pmf_seta * s_eta;
      et_i += pmf_seta * s_logtheta;
      tt_i += pmf * s_logtheta * s_logtheta;
    }

    double wi = w_pos(i);
    result.v_ee(i) = ee_i * wi;
    result.v_et(i) = et_i * wi;
    result.v_tt_sum += tt_i * wi;
  }

  return result;
}

// Assemble FIM matrix from components. Returns (kx+1) x (kx+1) matrix.
arma::mat assemble_fim(const ZtnbFimResult &comp, const arma::mat &X_pos) {
  int kx = X_pos.n_cols;
  int np = kx + 1;
  arma::mat fim(np, np, arma::fill::zeros);
  // Weighted crossproduct: X' diag(v_ee) X without allocating n_pos x n_pos
  arma::mat X_weighted = X_pos.each_col() % comp.v_ee;
  fim.submat(0, 0, kx - 1, kx - 1) = X_pos.t() * X_weighted;
  arma::vec xt_vet = X_pos.t() * comp.v_et;
  fim.submat(0, kx, kx - 1, kx) = xt_vet;
  fim.submat(kx, 0, kx, kx - 1) = xt_vet.t();
  fim(kx, kx) = comp.v_tt_sum;
  return fim;
}

// Convenience wrapper: returns the assembled FIM directly.
arma::mat compute_ztnb_fisher_info(const arma::vec &beta, double theta,
                                   const arma::mat &X_pos,
                                   const arma::vec &offset_pos,
                                   const arma::vec &w_pos,
                                   double quantile_cutoff = 0.9999) {
  auto comp = compute_ztnb_fim_components(beta, theta, X_pos, offset_pos, w_pos,
                                          quantile_cutoff);
  return assemble_fim(comp, X_pos);
}

// ==========================================================================
// Observed information (analytical negative Hessian) for ZTNB model
// ==========================================================================

// Compute the observed information matrix (negative Hessian of the ZTNB
// log-likelihood) analytically at given parameter values and observed data.
// Returns components in the same ZtnbFimResult structure as the expected FIM,
// so assemble_fim() can be reused directly.
//
// For logL_ZTNB = logf_NB(y) - log(1-p0), the observed info is:
//   I_obs = -d²logL_ZTNB/dθdθ' = I_NB + I_ZT
// where I_ZT uses: d²log(1-p0)/dudv = -r*a_uv - r*(1+r)*a_u*a_v
//
// Parameters are (beta_1, ..., beta_p, tau) where tau = log(theta).
// Cost: O(n_pos) per-observation loop (no PMF recurrence), plus O(n_pos*p²)
// for X'WX assembly via assemble_fim().
ZtnbFimResult compute_ztnb_observed_hessian_components(
    const arma::vec &beta, double theta, const arma::mat &X_pos,
    const arma::vec &offset_pos, const arma::vec &w_pos,
    const arma::vec &Y_pos) {
  int n_pos = X_pos.n_rows;
  double log_theta = std::log(theta);

  ZtnbFimResult result;
  result.v_ee.zeros(n_pos);
  result.v_et.zeros(n_pos);
  result.v_tt_sum = 0.0;

  for (int i = 0; i < n_pos; i++) {
    double eta_i = arma::dot(X_pos.row(i), beta) + offset_pos(i);
    double mu = std::exp(eta_i);
    double y = Y_pos(i);
    double A = mu + theta;           // A_i = mu_i + theta
    double A2 = A * A;
    double mu_over_A = mu / A;
    double theta_over_A = theta / A;

    // Zero-truncation quantities
    // log(p0) = -theta * log1p(mu/theta) for numerical stability at large theta
    double log_p0 = -theta * std::log1p(mu / theta);
    double p0 = std::exp(log_p0);
    double p1 = 1.0 - p0;
    if (p1 < 1e-300) p1 = 1e-300;
    double r = p0 / p1;              // r_i = p0/(1-p0)

    // Zero-truncation first derivatives of log(p0) w.r.t. (eta, theta):
    //   a_eta = d(log p0)/d(eta) = -theta*mu/A
    //   a_theta = d(log p0)/d(theta) = log(theta/A) + 1 - theta/A
    //           = -log1p(mu/theta) + mu/A   [stable at large theta]
    double a_eta = -theta * mu_over_A;
    double a_theta = -std::log1p(mu / theta) + mu_over_A;

    // Zero-truncation second derivatives of log(p0):
    //   a_eta_eta = d²(log p0)/d(eta)² = -theta*mu*theta/A²
    //            = -theta²*mu/A²
    double a_ee = -theta * theta * mu / A2;
    //   a_eta_theta = d²(log p0)/d(eta)d(theta) = -mu²/A²
    //              (d/dtheta of -theta*mu/A = -mu/A + theta*mu/A² = -mu²/A²... wait)
    //   Actually: a_eta = -theta*mu/A, so
    //   d(a_eta)/d(theta) = -mu/A + theta*mu/A² = -mu(A-theta)/A² = -mu²/A²
    double a_et = -mu * mu / A2;
    //   a_theta_theta = d²(log p0)/d(theta)² = mu²/(theta*A²)
    double a_tt = mu * mu / (theta * A2);

    // d²log(1-p0)/dudv = -r*a_uv - r*(1+r)*a_u*a_v
    // The observed info is I_obs = -d²logL_ZTNB = -d²logf_NB + d²(-log(1-p0))
    //                            = I_NB - d²log(1-p0)/dudv
    //                            = I_NB + r*a_uv + r*(1+r)*a_u*a_v
    // Wait: I_obs = -d²(logf_NB - log(1-p0)) = -d²logf_NB + d²log(1-p0)
    //             = I_NB + d²log(1-p0)/dudv
    //             = I_NB + (-r*a_uv - r*(1+r)*a_u*a_v)
    //             = I_NB - r*a_uv - r*(1+r)*a_u*a_v

    // ---- Beta-beta weight (w_ee) ----
    // I_NB_ee = mu*theta*(y+theta)/A²  [per obs, scalar multiplying x_j*x_k]
    double I_NB_ee = mu * theta * (y + theta) / A2;
    // ZT correction: -r*a_ee - r*(1+r)*a_eta²
    double ZT_ee = -r * a_ee - r * (1.0 + r) * a_eta * a_eta;
    result.v_ee(i) = w_pos(i) * (I_NB_ee + ZT_ee);

    // ---- Beta-logtheta weight (w_et) ----
    // Chain rule: d/d(tau) = theta * d/d(theta), so mixed derivative in (eta, tau):
    //   I_obs[eta, tau] = theta * I_obs[eta, theta]
    // I_NB[eta,theta] = mu*(mu-y)/A²
    double I_NB_et = mu * (mu - y) / A2;
    // ZT correction: -r*a_et - r*(1+r)*a_eta*a_theta
    double ZT_et = -r * a_et - r * (1.0 + r) * a_eta * a_theta;
    // Multiply by theta for tau = log(theta) parameterization
    result.v_et(i) = w_pos(i) * theta * (I_NB_et + ZT_et);

    // ---- Logtheta-logtheta weight (w_tt) ----
    // Chain rule: d²/d(tau)² = theta² * d²/d(theta)² + theta * d/d(theta)
    //
    // First compute d(logL_NB)/d(theta) / theta = b  (the "first derivative / theta")
    // Using recurrence for psi(y+theta) - psi(theta):
    //   digamma_diff = sum_{k=0}^{y-1} 1/(theta+k)
    double digamma_diff = 0.0;
    double trigamma_diff = 0.0;
    int yi = static_cast<int>(y);
    for (int k = 0; k < yi; k++) {
      double tk = theta + static_cast<double>(k);
      digamma_diff += 1.0 / tk;
      trigamma_diff += 1.0 / (tk * tk);
    }

    // b = digamma_diff + log(theta/A) + 1 - (y+theta)/A
    double log_ratio = log_theta - std::log(A);
    double b = digamma_diff + log_ratio + 1.0 - (y + theta) / A;

    // c = d²(logL_NB)/d(theta)² core
    //   = -trigamma_diff + 1/theta - 1/A + (y-mu)/A²
    // Note: trigamma_diff = psi1(theta) - psi1(y+theta) via recurrence
    double c = -trigamma_diff + 1.0 / theta - 1.0 / A + (y - mu) / A2;

    // I_NB[theta,theta] = -(b + theta*c)  [negative of d²logL/dtheta²]
    // Wait: d²logL_NB/dtheta² = d/dtheta(theta*b) = b + theta*c
    // So I_NB[theta,theta] = -(b + theta*c)  ... but we want observed INFO = -d²logL/d·²
    // Actually: d(logL_NB)/d(theta) = theta * b (approximately, from the structure)
    // Let me be precise. The NB loglik gradient w.r.t. theta is:
    //   d(logL_NB)/d(theta) = digamma_diff + log(theta/A) + 1 - (y+theta)/A = b
    // And d²(logL_NB)/d(theta)² = c (as defined above)
    // So I_NB[theta,theta] = -c

    // I_ZT[theta,theta] = r*a_tt + r*(1+r)*a_theta²
    double I_ZT_tt = r * a_tt + r * (1.0 + r) * a_theta * a_theta;

    // For tau = log(theta): d²/d(tau)² = theta² * d²/d(theta)² + theta * d/d(theta)
    // I_obs[tau,tau] = -d²logL/d(tau)² = -(theta² * d²logL/d(theta)² + theta * d(logL)/d(theta))
    //               = theta²*(-c - I_ZT_tt) + theta*(-b - r*a_theta) ... wait
    // More carefully:
    //   logL_ZTNB = logL_NB - log(1-p0)
    //   d(logL_ZTNB)/d(theta) = b + r*a_theta    [NB gradient + truncation correction]
    //   d²(logL_ZTNB)/d(theta)² = c + d/d(theta)[r*a_theta]
    //     d/d(theta)[r*a_theta] = (dr/dtheta)*a_theta + r*a_tt
    //     dr/dtheta = r*(1+r)*a_theta
    //     = r*(1+r)*a_theta² + r*a_tt = I_ZT_tt
    //   So d²(logL_ZTNB)/d(theta)² = c + I_ZT_tt
    //
    //   For tau: d(logL)/d(tau) = theta * [b + r*a_theta]
    //           d²(logL)/d(tau)² = theta*[b + r*a_theta] + theta²*[c + I_ZT_tt]
    //   I_obs[tau,tau] = -d²(logL)/d(tau)²
    //                  = -theta*(b + r*a_theta) - theta²*(c + I_ZT_tt)
    double first_deriv_theta = b + r * a_theta;
    double second_deriv_theta = c + I_ZT_tt;
    double w_tt_i = -theta * first_deriv_theta - theta * theta * second_deriv_theta;

    result.v_tt_sum += w_pos(i) * w_tt_i;
  }

  return result;
}

// Convenience wrapper: compute analytical observed Hessian, assembled.
arma::mat compute_ztnb_observed_info_analytical(
    const arma::vec &beta, double theta, const arma::mat &X_pos,
    const arma::vec &offset_pos, const arma::vec &w_pos,
    const arma::vec &Y_pos) {
  auto comp = compute_ztnb_observed_hessian_components(
      beta, theta, X_pos, offset_pos, w_pos, Y_pos);
  return assemble_fim(comp, X_pos);
}

// ==========================================================================
// Observed information (numerical Hessian) via finite differences — fallback
// ==========================================================================

// Compute the observed information matrix at a given parameter vector using
// central finite differences on the gradient. Works with any count functor.
// Returns an np x np symmetric matrix where np = parms.n_elem.
// The functor computes the gradient of -loglik, so -H = -(d(-grad)/dparms)
// = d(grad)/dparms, i.e., the observed info is the Jacobian of the
// negative-loglik gradient.
template <typename Functor>
arma::mat compute_observed_info(Functor &functor, const arma::vec &parms,
                                double eps_base = 1e-5) {
  int np = parms.n_elem;
  arma::mat H(np, np);

  // Central differences with parameter-scaled step size.
  // eps_j = eps_base * max(|parms_j|, 1) avoids too-small steps near zero
  // and scales with parameter magnitude for better numerical conditioning.
  for (int j = 0; j < np; j++) {
    double eps_j = eps_base * std::max(std::abs(parms(j)), 1.0);
    arma::vec parms_fwd = parms;
    arma::vec parms_bwd = parms;
    parms_fwd(j) += eps_j;
    parms_bwd(j) -= eps_j;

    arma::vec grad_fwd, grad_bwd;
    functor.Gradient(parms_fwd, grad_fwd);
    functor.Gradient(parms_bwd, grad_bwd);

    // grad = d(-logL)/dθ, so dgrad/dθ = d²(-logL)/dθdθ' = -d²logL/dθdθ'
    // That IS the observed information (negative Hessian of logL).
    H.col(j) = (grad_fwd - grad_bwd) / (2.0 * eps_j);
  }

  // Symmetrize (numerical noise can make it slightly asymmetric)
  return 0.5 * (H + H.t());
}

// R-exported test wrapper. Takes full data; filters to Y>0 internally.
// [[Rcpp::export]]
arma::mat compute_ztnb_fisher_info_cpp(const arma::vec &beta, double theta,
                                       const arma::mat &X,
                                       const arma::vec &offsetx,
                                       const arma::vec &weights) {
  // Filter to positive observations (FIM is only over Y>0)
  // For the test export, we assume all rows are Y>0 (caller pre-filters)
  return compute_ztnb_fisher_info(beta, theta, X, offsetx, weights);
}

// Count model Geometric functor (special case of Negative Binomial with theta =
// 1)
class CountGeomFunctor : public LikelihoodFunctor {
 private:
  // Use a shared_ptr to manage the CountNegBinFunctor instance
  std::shared_ptr<CountNegBinFunctor> negbin_functor;

 public:
  CountGeomFunctor(const arma::vec &y, const arma::mat &x,
                   const arma::vec &offs, const arma::vec &w)
      : LikelihoodFunctor(y, x, offs, w),
        negbin_functor(std::make_shared<CountNegBinFunctor>(y, x, offs, w)) {}

  double operator()(const arma::vec &parms) override {
    // Create a new parameter vector with an additional element for theta = 1
    // (log(theta) = 0)
    arma::vec parms_extended(parms.n_elem + 1);
    parms_extended.subvec(0, parms.n_elem - 1) = parms;
    parms_extended(parms.n_elem) = 0.0;  // log(1) = 0

    // Use the shared CountNegBinFunctor instance
    return (*negbin_functor)(parms_extended);
  }

  void Gradient(const arma::vec &parms, arma::vec &grad) override {
    // Create a new parameter vector with an additional element for theta = 1
    // (log(theta) = 0)
    arma::vec parms_extended(parms.n_elem + 1);
    parms_extended.subvec(0, parms.n_elem - 1) = parms;
    parms_extended(parms.n_elem) = 0.0;  // log(1) = 0

    // Use the shared CountNegBinFunctor instance for gradient calculation
    arma::vec grad_extended;
    negbin_functor->Gradient(parms_extended, grad_extended);

    // Return only the gradient for the original parameters (exclude theta)
    grad = arma::zeros<arma::vec>(parms.n_elem);
    grad.subvec(0, parms.n_elem - 1) =
        grad_extended.subvec(0, parms.n_elem - 1);
  }
};

// Zero hurdle Poisson functor
class ZeroPoissonFunctor : public LikelihoodFunctor {
 public:
  ZeroPoissonFunctor(const arma::vec &y, const arma::mat &x,
                     const arma::vec &offs, const arma::vec &w)
      : LikelihoodFunctor(y, x, offs, w) {}

  double operator()(const arma::vec &parms) override {
    // Calculate mu = exp(X * parms + offset)
    arma::vec eta = calculate_eta(parms);
    arma::vec mu = calculate_mu(eta);

    // Calculate log probability of zero: loglik0 = -mu
    arma::vec loglik0 = -mu;

    // Calculate log-likelihood
    double loglik = 0.0;

    // For Y=0 observations: sum(weights[Y0] * loglik0[Y0])
    if (Y0.n_elem > 0) {
      loglik += arma::dot(w_zero, loglik0.elem(Y0));
    }

    // For Y>0 observations: sum(weights[Y1] * log(1 - exp(loglik0[Y1])))
    if (Y1.n_elem > 0) {
      arma::vec temp = log1mexp(loglik0.elem(Y1));
      loglik += arma::dot(w_pos, temp);
    }

    // Return negative log-likelihood for minimization
    return -loglik;
  }

  void Gradient(const arma::vec &parms, arma::vec &grad) override {
    // Calculate eta = X * parms + offset
    arma::vec eta = calculate_eta(parms);

    // Calculate mu = exp(eta)
    arma::vec mu = calculate_mu(eta);

    // Initialize gradient term
    arma::vec grad_term = arma::zeros<arma::vec>(X.n_rows);

    // For Y=0 observations: -mu
    if (Y0.n_elem > 0) {
      grad_term.elem(Y0) = -mu.elem(Y0);
    }

    // For Y>0 observations: vectorized calculation
    if (Y1.n_elem > 0) {
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
class ZeroNegBinFunctor : public LikelihoodFunctor {
 public:
  ZeroNegBinFunctor(const arma::vec &y, const arma::mat &x,
                    const arma::vec &offs, const arma::vec &w)
      : LikelihoodFunctor(y, x, offs, w) {}

  double operator()(const arma::vec &parms) override {
    // Get number of parameters
    int kz = X.n_cols;

    // Calculate mu = exp(X * parms[0:kz-1] + offset) for all observations
    arma::vec mu_zero, mu_pos;
    double theta = exp(parms(kz));
    double log_theta = log(theta);

    // Calculate log-likelihood
    double loglik = 0.0;

    // For Y=0 observations
    if (Y0.n_elem > 0) {
      mu_zero = calculate_mu(X_zero * parms.subvec(0, kz - 1) + offset_zero);
      arma::vec loglik0_zero = theta * log_theta - theta * log(theta + mu_zero);
      loglik += arma::dot(w_zero, loglik0_zero);
    }

    // For Y>0 observations
    if (Y1.n_elem > 0) {
      mu_pos = calculate_mu(X_pos * parms.subvec(0, kz - 1) + offset_pos);
      arma::vec loglik0_pos = theta * log_theta - theta * log(theta + mu_pos);
      arma::vec temp = log1mexp(loglik0_pos);
      loglik += arma::dot(w_pos, temp);
    }

    // Return negative log-likelihood for minimization
    return -loglik;
  }

  void Gradient(const arma::vec &parms, arma::vec &grad) override {
    // Get number of parameters
    int kz = X.n_cols;

    double theta = exp(parms(kz));
    double log_theta = log(theta);

    // Initialize gradient for beta parameters
    arma::vec grad_beta = arma::zeros<arma::vec>(kz);
    double grad_logtheta = 0.0;

    // For Y=0 observations
    if (Y0.n_elem > 0) {
      arma::vec eta_0 = X_zero * parms.subvec(0, kz - 1) + offset_zero;
      arma::vec mu_0 = calculate_mu(eta_0);

      arma::vec term1 = -mu_0 * theta / (mu_0 + theta);
      grad_beta += X_zero.t() * (w_zero % term1);

      arma::vec mu_theta_0 = mu_0 + theta;
      arma::vec term_theta_0 =
          log_theta - log(mu_theta_0) + 1.0 - theta / mu_theta_0;
      grad_logtheta += arma::dot(w_zero, term_theta_0);
    }

    // For Y>0 observations
    if (Y1.n_elem > 0) {
      arma::vec eta_1 = X_pos * parms.subvec(0, kz - 1) + offset_pos;
      arma::vec mu_1 = calculate_mu(eta_1);

      arma::vec loglik0_1 = theta * log_theta - theta * log(theta + mu_1);
      arma::vec logratio = loglik0_1 - log1mexp(loglik0_1);
      arma::vec term2 = exp(logratio + log_theta - log(mu_1 + theta) + eta_1);
      grad_beta += X_pos.t() * (w_pos % term2);

      arma::vec mu_theta_1 = mu_1 + theta;
      arma::vec term_theta_1 = exp(logratio) % (log_theta - log(mu_theta_1) +
                                                1.0 - theta / mu_theta_1);
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

// Zero hurdle Geometric functor (special case of Negative Binomial with theta =
// 1)
class ZeroGeomFunctor : public LikelihoodFunctor {
 private:
  // Use a shared_ptr to manage the ZeroNegBinFunctor instance
  std::shared_ptr<ZeroNegBinFunctor> negbin_functor;

 public:
  ZeroGeomFunctor(const arma::vec &y, const arma::mat &x, const arma::vec &offs,
                  const arma::vec &w)
      : LikelihoodFunctor(y, x, offs, w),
        negbin_functor(std::make_shared<ZeroNegBinFunctor>(y, x, offs, w)) {}

  double operator()(const arma::vec &parms) override {
    // Create a new parameter vector with an additional element for theta = 1
    // (log(theta) = 0)
    arma::vec parms_extended(parms.n_elem + 1);
    parms_extended.subvec(0, parms.n_elem - 1) = parms;
    parms_extended(parms.n_elem) = 0.0;  // log(1) = 0

    // Use the shared ZeroNegBinFunctor instance
    return (*negbin_functor)(parms_extended);
  }

  void Gradient(const arma::vec &parms, arma::vec &grad) override {
    // Create a new parameter vector with an additional element for theta = 1
    // (log(theta) = 0)
    arma::vec parms_extended(parms.n_elem + 1);
    parms_extended.subvec(0, parms.n_elem - 1) = parms;
    parms_extended(parms.n_elem) = 0.0;  // log(1) = 0

    // Use the shared ZeroNegBinFunctor instance for gradient calculation
    arma::vec grad_extended;
    negbin_functor->Gradient(parms_extended, grad_extended);

    // Return only the gradient for the original parameters (exclude theta)
    grad = arma::zeros<arma::vec>(parms.n_elem);
    grad.subvec(0, parms.n_elem - 1) =
        grad_extended.subvec(0, parms.n_elem - 1);
  }
};

// Zero hurdle Binomial functor
class ZeroBinomFunctor : public LikelihoodFunctor {
 private:
  links::LinkFunction linkinv_func;
  links::LinkFunction mu_eta_func;
  std::string link_name;
  bool is_logit;  // Fast path for logit link

  // Cached binary indicator for logit fast path: 1.0 for Y>0, 0.0 for Y=0
  arma::vec Y_binary;

 public:
  ZeroBinomFunctor(const arma::vec &y, const arma::mat &x,
                   const arma::vec &offs, const arma::vec &w,
                   const std::string &link = "logit")
      : LikelihoodFunctor(y, x, offs, w),
        link_name(link),
        is_logit(link == "logit") {
    linkinv_func = links::get_linkinv(link);
    mu_eta_func = links::get_mu_eta(link);

    // For logit fast path, pre-compute binary indicator
    if (is_logit) {
      Y_binary = arma::conv_to<arma::vec>::from(Y > 0);
    }
  }

  double operator()(const arma::vec &parms) override {
    // Calculate eta = X * parms + offset
    arma::vec eta = calculate_eta(parms);

    if (is_logit) {
      // Logit fast path: use numerically stable softplus
      // log(sigmoid(x))  = -softplus(-x) where softplus(x) = log(1+exp(x))
      // log(1-sigmoid(x)) = -softplus(x)
      // Stabilized: softplus(x) = max(x,0) + log1p(exp(-|x|))
      double loglik = 0.0;
      if (Y1.n_elem > 0) {
        // -log(sigmoid(eta)) = softplus(-eta) = max(-eta,0) +
        // log1p(exp(-|-eta|))
        arma::vec eta_pos = eta.elem(Y1);
        arma::vec sp = arma::max(-eta_pos, arma::zeros(eta_pos.n_elem)) +
                       log1p(exp(-abs(eta_pos)));
        loglik -= arma::dot(w_pos, sp);
      }
      if (Y0.n_elem > 0) {
        // -log(1-sigmoid(eta)) = softplus(eta) = max(eta,0) +
        // log1p(exp(-|eta|))
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
    if (Y0.n_elem > 0) {
      arma::vec temp = log(1.0 - mu.elem(Y0));
      loglik += arma::dot(w_zero, temp);
    }
    if (Y1.n_elem > 0) {
      arma::vec temp = log(mu.elem(Y1));
      loglik += arma::dot(w_pos, temp);
    }
    return -loglik;
  }

  void Gradient(const arma::vec &parms, arma::vec &grad) override {
    // Calculate eta = X * parms + offset
    arma::vec eta = calculate_eta(parms);

    if (is_logit) {
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
    if (Y0.n_elem > 0) {
      grad_term.elem(Y0) = -1.0 / (1.0 - mu.elem(Y0));
    }
    if (Y1.n_elem > 0) {
      grad_term.elem(Y1) = 1.0 / mu.elem(Y1);
    }
    grad_term = grad_term % mu_eta_vec;

    grad = X.t() * (weights % grad_term);
    grad = -grad;
  }

  // Getter for link name
  std::string get_link_name() const { return link_name; }
};

// Combined functor for joint optimization
class JointFunctor : public Functor {
 private:
  std::shared_ptr<LikelihoodFunctor> count_functor;
  std::shared_ptr<LikelihoodFunctor> zero_functor;
  int kx;
  bool dist_negbin;

 public:
  JointFunctor(std::shared_ptr<LikelihoodFunctor> count,
               std::shared_ptr<LikelihoodFunctor> zero, int count_params,
               int zero_params, bool count_is_negbin, bool zero_is_negbin)
      : count_functor(count),
        zero_functor(zero),
        kx(count_params),
        dist_negbin(count_is_negbin) {
    // Suppress unused parameter warnings
    (void)zero_params;
    (void)zero_is_negbin;
  }

  double operator()(const arma::vec &parms) override {
    // Split parameters for count and zero components
    arma::vec count_parms = parms.subvec(0, kx + (dist_negbin ? 0 : -1));
    arma::vec zero_parms =
        parms.subvec(kx + (dist_negbin ? 1 : 0), parms.n_elem - 1);

    // Calculate log-likelihood for both components
    double count_loglik = (*count_functor)(count_parms);
    double zero_loglik = (*zero_functor)(zero_parms);

    // Return combined negative log-likelihood
    return count_loglik + zero_loglik;
  }

  void Gradient(const arma::vec &parms, arma::vec &grad) override {
    // Split parameters for count and zero components
    arma::vec count_parms = parms.subvec(0, kx + (dist_negbin ? 0 : -1));
    arma::vec zero_parms =
        parms.subvec(kx + (dist_negbin ? 1 : 0), parms.n_elem - 1);

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
Rcpp::List run_optimization(FunctorType &functor, arma::vec &start,
                            const std::string &method = "BFGS",
                            bool hessian = true, int maxit = 10000,
                            double reltol = -1.0) {
  // Create optimizer
  Roptim<FunctorType> opt(method);
  opt.control.trace = 0;
  opt.control.maxit = maxit;
  if (reltol > 0.0) {
    opt.control.reltol = reltol;
  }
  opt.set_hessian(hessian);

  // Optimize
  opt.minimize(functor, start);

  // Return results
  return Rcpp::List::create(
      Rcpp::Named("par") = opt.par(),
      Rcpp::Named("value") = -opt.value(),  // Convert back to log-likelihood
      Rcpp::Named("counts") =
          Rcpp::List::create(Rcpp::Named("function") = opt.fncount(),
                             Rcpp::Named("gradient") = opt.grcount()),
      Rcpp::Named("convergence") = opt.convergence(),
      Rcpp::Named("message") = opt.message(),
      Rcpp::Named("hessian") = opt.hessian());
}

// R interface functions for optimization using roptim

// [[Rcpp::export]]
Rcpp::List optim_count_poisson_cpp(const arma::vec &start, const arma::vec &Y,
                                   const arma::mat &X, const arma::vec &offsetx,
                                   const arma::vec &weights,
                                   const std::string &method = "BFGS",
                                   bool hessian = true, int maxit = 10000,
                                   double reltol = -1.0) {
  // Create functor
  CountPoissonFunctor functor(Y, X, offsetx, weights);

  // Run optimization
  arma::vec par = start;
  return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List optim_count_negbin_cpp(const arma::vec &start, const arma::vec &Y,
                                  const arma::mat &X, const arma::vec &offsetx,
                                  const arma::vec &weights,
                                  const std::string &method = "BFGS",
                                  bool hessian = true, int maxit = 10000,
                                  double reltol = -1.0) {
  CountNegBinFunctor functor(Y, X, offsetx, weights);
  arma::vec par = start;
  return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List optim_count_geom_cpp(const arma::vec &start, const arma::vec &Y,
                                const arma::mat &X, const arma::vec &offsetx,
                                const arma::vec &weights,
                                const std::string &method = "BFGS",
                                bool hessian = true, int maxit = 10000,
                                double reltol = -1.0) {
  // Create functor
  CountGeomFunctor functor(Y, X, offsetx, weights);

  // Run optimization
  arma::vec par = start;
  return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List optim_zero_poisson_cpp(const arma::vec &start, const arma::vec &Y,
                                  const arma::mat &X, const arma::vec &offsetx,
                                  const arma::vec &weights,
                                  const std::string &method = "BFGS",
                                  bool hessian = true, int maxit = 10000,
                                  double reltol = -1.0) {
  // Create functor
  ZeroPoissonFunctor functor(Y, X, offsetx, weights);

  // Run optimization
  arma::vec par = start;
  return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List optim_zero_negbin_cpp(const arma::vec &start, const arma::vec &Y,
                                 const arma::mat &X, const arma::vec &offsetx,
                                 const arma::vec &weights,
                                 const std::string &method = "BFGS",
                                 bool hessian = true, int maxit = 10000,
                                 double reltol = -1.0) {
  // Create functor
  ZeroNegBinFunctor functor(Y, X, offsetx, weights);

  // Run optimization
  arma::vec par = start;
  return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List optim_zero_geom_cpp(const arma::vec &start, const arma::vec &Y,
                               const arma::mat &X, const arma::vec &offsetx,
                               const arma::vec &weights,
                               const std::string &method = "BFGS",
                               bool hessian = true, int maxit = 10000,
                               double reltol = -1.0) {
  // Create functor
  ZeroGeomFunctor functor(Y, X, offsetx, weights);

  // Run optimization
  arma::vec par = start;
  return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List optim_zero_binom_cpp(const arma::vec &start, const arma::vec &Y,
                                const arma::mat &X, const arma::vec &offsetx,
                                const arma::vec &weights,
                                const std::string &link = "logit",
                                const std::string &method = "BFGS",
                                bool hessian = true, int maxit = 10000,
                                double reltol = -1.0) {
  // Create functor with C++ link function
  ZeroBinomFunctor functor(Y, X, offsetx, weights, link);

  // Run optimization
  arma::vec par = start;
  return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// [[Rcpp::export]]
Rcpp::List optim_joint_cpp(
    const arma::vec &start, const arma::vec &Y, const arma::mat &X,
    const arma::vec &offsetx, const arma::mat &Z, const arma::vec &offsetz,
    const arma::vec &weights, const std::string &dist = "poisson",
    const std::string &zero_dist = "binomial",
    const std::string &link = "logit", const std::string &method = "BFGS",
    bool hessian = true, int maxit = 10000, double reltol = -1.0) {
  // Create count functor based on distribution
  std::shared_ptr<LikelihoodFunctor> count_functor;
  bool dist_negbin = false;

  if (dist == "poisson") {
    count_functor =
        std::make_shared<CountPoissonFunctor>(Y, X, offsetx, weights);
  } else if (dist == "negbin") {
    count_functor =
        std::make_shared<CountNegBinFunctor>(Y, X, offsetx, weights);
    dist_negbin = true;
  } else if (dist == "geometric") {
    count_functor = std::make_shared<CountGeomFunctor>(Y, X, offsetx, weights);
  } else {
    Rcpp::stop("Unknown count distribution");
  }

  // Create zero functor based on distribution
  std::shared_ptr<LikelihoodFunctor> zero_functor;
  bool zero_dist_negbin = false;

  if (zero_dist == "poisson") {
    zero_functor = std::make_shared<ZeroPoissonFunctor>(Y, Z, offsetz, weights);
  } else if (zero_dist == "negbin") {
    zero_functor = std::make_shared<ZeroNegBinFunctor>(Y, Z, offsetz, weights);
    zero_dist_negbin = true;
  } else if (zero_dist == "geometric") {
    zero_functor = std::make_shared<ZeroGeomFunctor>(Y, Z, offsetz, weights);
  } else if (zero_dist == "binomial") {
    zero_functor =
        std::make_shared<ZeroBinomFunctor>(Y, Z, offsetz, weights, link);
  } else {
    Rcpp::stop("Unknown zero hurdle distribution");
  }

  // Create joint functor
  JointFunctor functor(count_functor, zero_functor, X.n_cols, Z.n_cols,
                       dist_negbin, zero_dist_negbin);

  // Run optimization
  arma::vec par = start;
  return run_optimization(functor, par, method, hessian, maxit, reltol);
}

// ==========================================================================
// Saddlepoint approximation (SPA) for score test p-values
// ==========================================================================

// CGF result: K(t), K'(t), K''(t) for the adjusted score Σ s_eta_i * g_i
struct ZtnbCgfResult {
  double K;   // CGF value
  double K1;  // First derivative
  double K2;  // Second derivative
};

// Pre-computed per-observation constants for closed-form CGF evaluation.
// The ZTNB score s_eta(y) = alpha*y - C is linear in y, so the CGF
// uses the NB MGF in closed form — no PMF summation needed.
struct ZtnbCgfObsCache {
  double mu;
  double alpha;       // theta / (mu + theta)
  double C_i;         // alpha * mu * (1 + r), where r = p0/(1-p0)
  double p0;          // (theta/(mu+theta))^theta
  double log_p1;      // log(1 - p0)
  double gi;          // covariate-projected test variable
  double wi;          // weight
  double lambda_max;  // log(1 + theta/mu) = MGF domain bound
};

// Build per-observation cache for closed-form CGF.
std::vector<ZtnbCgfObsCache> build_cgf_cache(const arma::vec &beta,
                                             double theta,
                                             const arma::mat &X_pos,
                                             const arma::vec &offset_pos,
                                             const arma::vec &w_pos,
                                             const arma::vec &g_tilde_pos) {
  int n_pos = X_pos.n_rows;
  double log_theta = std::log(theta);
  std::vector<ZtnbCgfObsCache> cache;
  cache.reserve(n_pos);

  for (int i = 0; i < n_pos; i++) {
    double gi = g_tilde_pos(i);
    if (std::abs(gi) < 1e-15) continue;

    double eta_i = arma::dot(X_pos.row(i), beta) + offset_pos(i);
    double mu = std::exp(eta_i);
    double mu_theta = mu + theta;
    double alpha = theta / mu_theta;

    double log_p0 = theta * (log_theta - std::log(mu_theta));
    double p0 = std::exp(log_p0);
    // Stable log(1-p0): use log(-expm1(log_p0)) when p0 > 0.5
    double log_p1 =
        (p0 > 0.5) ? std::log(-std::expm1(log_p0)) : std::log1p(-p0);
    double p1 = std::exp(log_p1);
    if (p1 < 1e-300) {
      // p1 ≈ 0 means this obs has degenerate ZTNB — SPA cannot be used
      cache.clear();
      return cache;  // empty cache signals SPA failure
    }
    // r = p0/p1 in log space for stability
    double r = std::exp(log_p0 - log_p1);

    ZtnbCgfObsCache obs;
    obs.mu = mu;
    obs.alpha = alpha;
    obs.C_i = alpha * mu * (1.0 + r);
    obs.p0 = p0;
    obs.log_p1 = log_p1;
    obs.gi = gi;
    obs.wi = w_pos(i);
    obs.lambda_max = std::log(1.0 + theta / mu);
    cache.push_back(obs);
  }
  return cache;
}

// Evaluate closed-form CGF at t. No PMF loop — O(1) per observation.
//
// K_i(t) = -C_i*g_i*t + log(M_NB(lambda) - p0) - log(1-p0)
// where lambda = alpha_i * g_i * t, M_NB(lambda) = (theta/D)^theta,
// D = theta + mu*(1 - exp(lambda))
// Return value signaling CGF failure (NaN propagation triggers fallback)
static const ZtnbCgfResult CGF_FAILURE = {
    std::numeric_limits<double>::quiet_NaN(),
    std::numeric_limits<double>::quiet_NaN(),
    std::numeric_limits<double>::quiet_NaN()};

ZtnbCgfResult compute_ztnb_cgf_cached(
    double t, double theta, const std::vector<ZtnbCgfObsCache> &cache) {
  ZtnbCgfResult result = {0.0, 0.0, 0.0};

  for (const auto &obs : cache) {
    double lambda = obs.alpha * obs.gi * t;

    // Check MGF domain: need D > 0, i.e., lambda < log(1 + theta/mu)
    if (lambda >= obs.lambda_max * 0.999) return CGF_FAILURE;

    double el = std::exp(lambda);
    double D = theta + obs.mu * (1.0 - el);
    if (D <= 1e-300) return CGF_FAILURE;

    // M_NB = (theta/D)^theta
    double log_M = theta * (std::log(theta) - std::log(D));
    double M = std::exp(log_M);
    double M_minus_p0 = M - obs.p0;

    // Numerical stability: when M ≈ p0 (small |lambda|), use expm1 form
    // M/p0 = ((theta+mu)/D)^theta, so M/p0 - 1 = expm1(theta*log((theta+mu)/D))
    double log_M_minus_p0;
    if (std::abs(M_minus_p0) < obs.p0 * 1e-6) {
      double log_ratio = theta * std::log((theta + obs.mu) / D);
      double ratio = std::expm1(log_ratio);  // expm1 for stability near 0
      if (ratio <= 0) return CGF_FAILURE;
      log_M_minus_p0 = std::log(obs.p0) + std::log1p(ratio - 1.0 + 1.0);
      // simplify: log(p0) + log(ratio) since ratio = expm1(x) > 0
      log_M_minus_p0 = std::log(obs.p0) + std::log(ratio);
    } else {
      if (M_minus_p0 <= 0) return CGF_FAILURE;
      log_M_minus_p0 = std::log(M_minus_p0);
    }

    // K_i(t) = -C*g*t + log(M-p0) - log(p1)
    double K_i = -obs.C_i * obs.gi * t + log_M_minus_p0 - obs.log_p1;

    // Derivatives via h = theta * mu * exp(lambda) / D
    double mu_el = obs.mu * el;
    double h = theta * mu_el / D;

    // M' = dM/dlambda = h * M
    double M1 = h * M;

    // R = M1 / (M - p0) = h * M / (M - p0)
    double R = M1 / M_minus_p0;

    // K_i'(t) = alpha*g * (-C/(alpha) + R) = alpha*g*R - C*g
    double ag = obs.alpha * obs.gi;
    double K1_i = ag * R - obs.C_i * obs.gi;

    // M'' = M * h * (h + 1 + mu*exp(lambda)/D)
    // K_i''(t) = (alpha*g)^2 * [M''/(M-p0) - R^2]
    //          = (alpha*g)^2 * [h*(h + 1 + mu_el/D) * M/(M-p0) - R^2]
    //          = (alpha*g)^2 * [R*(h + 1 + mu_el/D) - R^2]
    //          = (alpha*g)^2 * R * (h + 1 + mu_el/D - R)
    double K2_i = ag * ag * R * (h + 1.0 + mu_el / D - R);

    result.K += obs.wi * K_i;
    result.K1 += obs.wi * K1_i;
    result.K2 += obs.wi * K2_i;
  }

  return result;
}

// Find saddlepoint: solve K'(zeta) = q using Newton's method with bisection
// fallback. Uses pre-computed cache for efficiency.
struct SaddlepointResult {
  double zeta;
  bool converged;
};

SaddlepointResult find_saddlepoint(double q, double theta,
                                   const std::vector<ZtnbCgfObsCache> &cache,
                                   double tol = 1e-8, int maxiter = 100) {
  SaddlepointResult res = {0.0, false};
  double t = 0.0;

  auto cgf = compute_ztnb_cgf_cached(t, theta, cache);
  double K1_eval = cgf.K1 - q;
  double K2_eval = cgf.K2;
  double prev_jump = std::numeric_limits<double>::infinity();

  for (int iter = 0; iter < maxiter; iter++) {
    if (K2_eval < 1e-20) K2_eval = 1e-20;
    double tnew = t - K1_eval / K2_eval;

    if (std::abs(tnew - t) < tol) {
      res.zeta = tnew;
      res.converged = true;
      return res;
    }

    cgf = compute_ztnb_cgf_cached(tnew, theta, cache);
    double new_K1 = cgf.K1 - q;

    if (std::isnan(tnew) || std::isnan(new_K1)) break;

    // Bisection safeguard (from SAIGE)
    if ((K1_eval > 0) != (new_K1 > 0)) {
      if (std::abs(tnew - t) > (prev_jump - tol)) {
        tnew = t + ((new_K1 > K1_eval) ? 1.0 : -1.0) * prev_jump / 2.0;
        cgf = compute_ztnb_cgf_cached(tnew, theta, cache);
        new_K1 = cgf.K1 - q;
        prev_jump = prev_jump / 2.0;
      } else {
        prev_jump = std::abs(tnew - t);
      }
    }

    t = tnew;
    K1_eval = new_K1;
    K2_eval = cgf.K2;
  }

  res.zeta = t;
  res.converged = false;
  return res;
}

// Compute SPA p-value using Lugannani-Rice formula (one-sided tail prob)
double spa_pvalue_one_tail(double zeta, double q, double theta,
                           const std::vector<ZtnbCgfObsCache> &cache) {
  auto cgf = compute_ztnb_cgf_cached(zeta, theta, cache);

  double temp1 = zeta * q - cgf.K;
  if (!std::isfinite(cgf.K) || !std::isfinite(cgf.K2) || temp1 < 0 ||
      cgf.K2 < 0) {
    return -1.0;  // signal failure
  }

  double w = (zeta > 0 ? 1.0 : -1.0) * std::sqrt(2.0 * temp1);
  double v = zeta * std::sqrt(cgf.K2);

  if (std::abs(w) < 1e-15) return -1.0;

  double z_spa = w + (1.0 / w) * std::log(v / w);
  if (std::isnan(z_spa)) return -1.0;

  if (z_spa > 0) {
    return R::pnorm(z_spa, 0.0, 1.0, 0, 0);  // upper tail
  } else {
    return R::pnorm(z_spa, 0.0, 1.0, 1, 0);  // lower tail
  }
}

// Full two-sided SPA p-value. Falls back to pval_nospa on failure.
// Takes pre-built cache; build it once with build_cgf_cache() before calling.
double spa_pvalue_twosided(double q, double pval_nospa, double theta,
                           const std::vector<ZtnbCgfObsCache> &cache,
                           double tol = 1e-8) {
  auto sp1 = find_saddlepoint(q, theta, cache, tol);
  auto sp2 = find_saddlepoint(-q, theta, cache, tol);

  if (!sp1.converged || !sp2.converged) return pval_nospa;

  double p1 = spa_pvalue_one_tail(sp1.zeta, q, theta, cache);
  double p2 = spa_pvalue_one_tail(sp2.zeta, -q, theta, cache);

  if (p1 < 0 || p2 < 0) return pval_nospa;

  return std::abs(p1) + std::abs(p2);
}

// ==========================================================================
// Score test for count component
// ==========================================================================

// [[Rcpp::export]]
Rcpp::List score_test_count_cpp(const arma::vec &null_par, const arma::vec &Y,
                                const arma::mat &X_null,
                                const arma::mat &X_full,
                                const arma::vec &offsetx,
                                const arma::vec &weights,
                                const std::string &dist = "negbin",
                                bool use_spa = false, double spa_cutoff = 2.0) {
  int kx_null = X_null.n_cols;
  int kx_full = X_full.n_cols;
  int n_test = kx_full - kx_null;

  // Build full-model parameter vector at the null MLE:
  // Insert 0 for the test variable(s) at position kx_null
  arma::vec parms_full;
  if (dist == "negbin") {
    parms_full.zeros(kx_full + 1);
    parms_full.subvec(0, kx_null - 1) = null_par.subvec(0, kx_null - 1);
    parms_full(kx_full) = null_par(kx_null);  // logtheta
  } else {
    parms_full.zeros(kx_full);
    parms_full.subvec(0, kx_null - 1) = null_par;
  }

  // Evaluate gradient at the null (using full model design matrix)
  // Functors return gradient of -loglik, so score = -grad
  arma::vec grad;
  if (dist == "negbin") {
    CountNegBinFunctor functor(Y, X_full, offsetx, weights);
    functor.Gradient(parms_full, grad);
  } else if (dist == "poisson") {
    CountPoissonFunctor functor(Y, X_full, offsetx, weights);
    functor.Gradient(parms_full, grad);
  } else if (dist == "geometric") {
    CountGeomFunctor functor(Y, X_full, offsetx, weights);
    functor.Gradient(parms_full, grad);
  } else {
    Rcpp::stop("Unsupported dist: " + dist);
  }

  arma::vec U_test = -grad.subvec(kx_null, kx_full - 1);

  // Compute expected FIM at the null for the full model
  arma::uvec Y1 = arma::find(Y > 0);
  if (Y1.n_elem == 0) {
    return Rcpp::List::create(
        Rcpp::Named("beta") = arma::vec(n_test, arma::fill::zeros),
        Rcpp::Named("se") = arma::vec(n_test, arma::fill::value(R_PosInf)),
        Rcpp::Named("statistic") = 0.0, Rcpp::Named("pvalue") = 1.0,
        Rcpp::Named("spa_applied") = false);
  }

  // Compute information matrix at the null MLE.
  // Try observed information (negative Hessian) first — more robust under
  // model misspecification. Fall back to expected FIM if observed info is
  // indefinite (e.g., small n_pos, boundary theta).
  double theta_fim;
  arma::vec beta_full = parms_full.subvec(0, kx_full - 1);
  if (dist == "negbin") {
    theta_fim = std::exp(parms_full(kx_full));
  } else if (dist == "geometric") {
    theta_fim = 1.0;
  } else {
    theta_fim = 1e8;  // Poisson ≈ NB with large theta
  }

  arma::mat fim;
  bool used_observed_info = false;

  // Observed information via numerical differentiation on the gradient.
  // IMPORTANT: Use only Y>0 observations — the count model in the hurdle
  // operates on positive counts only. The functor subsets internally, but
  // we must pass only Y>0 data so the Hessian reflects the count component.
  {
    arma::vec Y_pos = Y.elem(Y1);
    arma::mat X_pos = X_full.rows(Y1);
    arma::vec off_pos = offsetx.elem(Y1);
    arma::vec w_pos = weights.elem(Y1);

    if (dist == "negbin") {
      // Analytical observed Hessian for NB (no finite differences needed)
      fim = compute_ztnb_observed_info_analytical(beta_full, theta_fim,
                                                   X_pos, off_pos, w_pos, Y_pos);
      used_observed_info = true;
    } else if (dist == "poisson") {
      CountPoissonFunctor obs_functor(Y_pos, X_pos, off_pos, w_pos);
      fim = compute_observed_info(obs_functor, parms_full);
      used_observed_info = true;
    } else if (dist == "geometric") {
      CountGeomFunctor obs_functor(Y_pos, X_pos, off_pos, w_pos);
      fim = compute_observed_info(obs_functor, parms_full);
      used_observed_info = true;
    }
  }

  // Validate observed Hessian: must be finite. We do NOT require the full
  // matrix to be PD — the full observed info can be indefinite while the
  // Schur complement (effective information for the test variable) is still
  // positive. The Schur complement positivity is checked separately below.
  bool obs_info_valid = used_observed_info && fim.is_finite();
  if (!obs_info_valid) {
    // Fall back to expected FIM. For Poisson/geometric, the expected FIM
    // uses theta_fim (1e8 or 1.0) and returns (kx_full+1) x (kx_full+1),
    // which is the same behavior as before observed info was added.
    fim = compute_ztnb_fisher_info(beta_full, theta_fim, X_full.rows(Y1),
                                   offsetx.elem(Y1), weights.elem(Y1));
    used_observed_info = false;
  }

  // Extract FIM blocks for Schur complement.
  // Project out all nuisance parameters (null betas + theta for NB).
  // I_test_eff = I_tt - I_tn * I_nn^{-1} * I_nt
  int np_fim = fim.n_rows;  // kx_full+1 for NB, kx_full for Poisson/Geom
  // Nuisance indices: 0..kx_null-1 and kx_full..np_fim-1 (theta if NB)
  arma::uvec null_idx, test_idx;
  test_idx = arma::regspace<arma::uvec>(kx_null, kx_full - 1);
  if (np_fim > kx_full) {
    // NB: null = covariate betas + theta
    null_idx = arma::join_cols(arma::regspace<arma::uvec>(0, kx_null - 1),
                               arma::regspace<arma::uvec>(kx_full, np_fim - 1));
  } else {
    null_idx = arma::regspace<arma::uvec>(0, kx_null - 1);
  }
  arma::mat I_nn = fim(null_idx, null_idx);
  arma::mat I_nt = fim(null_idx, test_idx);
  arma::mat I_tt = fim(test_idx, test_idx);

  // Guard solve for singular I_nn (computation failure → return NA)
  arma::mat I_nn_inv_Int;
  bool count_solve_ok = arma::solve(I_nn_inv_Int, I_nn, I_nt);
  if (!count_solve_ok) {
    return Rcpp::List::create(
        Rcpp::Named("beta") = arma::vec(n_test, arma::fill::value(NA_REAL)),
        Rcpp::Named("se") = arma::vec(n_test, arma::fill::value(NA_REAL)),
        Rcpp::Named("statistic") = NA_REAL, Rcpp::Named("pvalue") = NA_REAL,
        Rcpp::Named("spa_applied") = false);
  }
  arma::mat I_test = I_tt - I_nt.t() * I_nn_inv_Int;

  // Guard: Schur complement must be positive (definite for n_test > 1).
  // If observed info produced a non-positive I_test, return NA.
  if (n_test == 1 && (I_test(0, 0) <= 0 || !std::isfinite(I_test(0, 0)))) {
    return Rcpp::List::create(
        Rcpp::Named("beta") = arma::vec(n_test, arma::fill::value(NA_REAL)),
        Rcpp::Named("se") = arma::vec(n_test, arma::fill::value(NA_REAL)),
        Rcpp::Named("statistic") = NA_REAL, Rcpp::Named("pvalue") = NA_REAL,
        Rcpp::Named("spa_applied") = false);
  }

  // Score test statistic: T = U' I_eff^{-1} U ~ chi2(n_test)
  double T_stat;
  arma::vec beta_hat(n_test);  // ratio estimator: beta = U / I

  if (n_test == 1) {
    double I_val = I_test(0, 0);
    T_stat = U_test(0) * U_test(0) / I_val;
    beta_hat(0) = U_test(0) / I_val;
  } else {
    arma::mat I_test_inv;
    bool test_solve_ok =
        arma::solve(I_test_inv, I_test, arma::eye(n_test, n_test));
    if (!test_solve_ok) {
      return Rcpp::List::create(
          Rcpp::Named("beta") = arma::vec(n_test, arma::fill::value(NA_REAL)),
          Rcpp::Named("se") = arma::vec(n_test, arma::fill::value(NA_REAL)),
          Rcpp::Named("statistic") = NA_REAL, Rcpp::Named("pvalue") = NA_REAL,
          Rcpp::Named("spa_applied") = false);
    }
    T_stat = arma::as_scalar(U_test.t() * I_test_inv * U_test);
    beta_hat = I_test_inv * U_test;
  }

  // p-value: chi-squared approximation (baseline)
  double pvalue = R::pchisq(T_stat, static_cast<double>(n_test), 0, 0);

  // SPA: if requested and |z| > cutoff, refine p-value using saddlepoint
  bool spa_applied = false;
  if (use_spa && n_test == 1 && std::sqrt(T_stat) > spa_cutoff) {
    // Compute g_tilde: covariate-projected test variable
    arma::mat I_nn_beta = fim.submat(0, 0, kx_null - 1, kx_null - 1);
    arma::vec I_nt_beta = fim.submat(0, kx_null, kx_null - 1, kx_null);
    arma::vec proj_coef = arma::solve(I_nn_beta, I_nt_beta);
    arma::vec g_tilde =
        X_full.col(kx_null) - X_full.cols(0, kx_null - 1) * proj_coef;
    arma::vec g_tilde_pos = g_tilde.elem(Y1);

    // Build per-observation cache once, reused across all Newton iterations
    // Empty cache signals a degenerate observation — skip SPA entirely
    auto cgf_cache =
        build_cgf_cache(beta_full, theta_fim, X_full.rows(Y1), offsetx.elem(Y1),
                        weights.elem(Y1), g_tilde_pos);

    if (!cgf_cache.empty()) {
      double p_spa =
          spa_pvalue_twosided(U_test(0), pvalue, theta_fim, cgf_cache);
      if (p_spa >= 0 && p_spa <= 1.0) {
        pvalue = p_spa;
        spa_applied = true;
      }
    }
  }

  // Beta refinement: run a few BFGS iterations from the score estimate
  // to correct the ratio estimator bias. With observed information, the
  // ratio estimator (U/I_eff) can be more biased than with expected FIM,
  // so we trigger refinement when |z| exceeds the cutoff (default 2.0).
  // When SPA is enabled, use spa_cutoff; otherwise default to 2.0.
  // Accept refined beta only if finite and objective improved.
  double refine_cutoff = use_spa ? spa_cutoff : 2.0;
  bool refine_beta =
      (n_test == 1) && (std::sqrt(T_stat) > refine_cutoff);
  if (refine_beta) {
    arma::vec start_refine;
    if (dist == "negbin") {
      start_refine.zeros(kx_full + 1);
      start_refine.subvec(0, kx_null - 1) = null_par.subvec(0, kx_null - 1);
      start_refine(kx_null) = beta_hat(0);
      start_refine(kx_full) = null_par(kx_null);  // logtheta
    } else {
      start_refine.zeros(kx_full);
      start_refine.subvec(0, kx_null - 1) = null_par;
      start_refine(kx_null) = beta_hat(0);
    }

    // Evaluate objective at starting point
    double obj_before;
    if (dist == "negbin") {
      CountNegBinFunctor f0(Y, X_full, offsetx, weights);
      obj_before = f0(start_refine);
    } else if (dist == "poisson") {
      CountPoissonFunctor f0(Y, X_full, offsetx, weights);
      obj_before = f0(start_refine);
    } else {
      CountGeomFunctor f0(Y, X_full, offsetx, weights);
      obj_before = f0(start_refine);
    }

    // Run 5-iteration BFGS refinement
    double obj_after = obj_before;
    double beta_refined = beta_hat(0);
    if (dist == "negbin") {
      CountNegBinFunctor functor(Y, X_full, offsetx, weights);
      arma::vec par = start_refine;
      Roptim<CountNegBinFunctor> opt("BFGS");
      opt.control.trace = 0;
      opt.control.maxit = 5;
      opt.set_hessian(false);
      opt.minimize(functor, par);
      obj_after = opt.value();
      beta_refined = opt.par()(kx_null);
    } else if (dist == "poisson") {
      CountPoissonFunctor functor(Y, X_full, offsetx, weights);
      arma::vec par = start_refine;
      Roptim<CountPoissonFunctor> opt("BFGS");
      opt.control.trace = 0;
      opt.control.maxit = 5;
      opt.set_hessian(false);
      opt.minimize(functor, par);
      obj_after = opt.value();
      beta_refined = opt.par()(kx_null);
    } else {
      CountGeomFunctor functor(Y, X_full, offsetx, weights);
      arma::vec par = start_refine;
      Roptim<CountGeomFunctor> opt("BFGS");
      opt.control.trace = 0;
      opt.control.maxit = 5;
      opt.set_hessian(false);
      opt.minimize(functor, par);
      obj_after = opt.value();
      beta_refined = opt.par()(kx_null);
    }

    // Accept only if finite and objective improved (or equal)
    if (std::isfinite(beta_refined) && std::isfinite(obj_after) &&
        obj_after <= obj_before) {
      beta_hat(0) = beta_refined;
    }
  }

  // SE: back-computed from p-value to keep beta/SE/p consistent.
  arma::vec se_hat(n_test);
  if (pvalue > 0.0 && pvalue < 1.0) {
    double z_abs = R::qnorm(pvalue / 2.0, 0.0, 1.0, 0, 0);
    for (int j = 0; j < n_test; j++) {
      se_hat(j) = std::abs(beta_hat(j)) / z_abs;
    }
  } else {
    for (int j = 0; j < n_test; j++) {
      se_hat(j) = (pvalue == 0.0) ? 0.0 : R_PosInf;
    }
  }

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_hat, Rcpp::Named("se") = se_hat,
      Rcpp::Named("statistic") = T_stat, Rcpp::Named("pvalue") = pvalue,
      Rcpp::Named("spa_applied") = spa_applied);
}

// ==========================================================================
// Score test for zero (binomial/logit) component
// ==========================================================================

// Binomial SPA: per-observation cache (closed-form, logit only)
struct BinomCgfObsCache {
  double pi;  // logistic(eta_null)
  double gi;  // covariate-projected test variable
  double wi;  // weight
};

// Mean-centered binomial CGF: K_i(t) = -p*g*t + log(1-p + p*exp(g*t))
struct BinomCgfResult {
  double K, K1, K2;
};

BinomCgfResult compute_binom_cgf(double t,
                                 const std::vector<BinomCgfObsCache> &cache) {
  BinomCgfResult result = {0.0, 0.0, 0.0};
  for (const auto &obs : cache) {
    double gt = obs.gi * t;
    double egt = std::exp(gt);
    double denom = 1.0 - obs.pi + obs.pi * egt;
    if (denom < 1e-300) continue;

    double K_i = -obs.pi * obs.gi * t + std::log(denom);
    double p_egt_over_denom = obs.pi * egt / denom;
    double K1_i = obs.gi * (p_egt_over_denom - obs.pi);
    double K2_i =
        obs.pi * (1.0 - obs.pi) * obs.gi * obs.gi * egt / (denom * denom);

    result.K += obs.wi * K_i;
    result.K1 += obs.wi * K1_i;
    result.K2 += obs.wi * K2_i;
  }
  return result;
}

// Binomial saddlepoint solver (same Newton + bisection as count)
SaddlepointResult find_saddlepoint_binom(
    double q, const std::vector<BinomCgfObsCache> &cache, double tol = 1e-8,
    int maxiter = 100) {
  SaddlepointResult res = {0.0, false};
  double t = 0.0;

  auto cgf = compute_binom_cgf(t, cache);
  double K1_eval = cgf.K1 - q;
  double K2_eval = cgf.K2;
  double prev_jump = std::numeric_limits<double>::infinity();

  for (int iter = 0; iter < maxiter; iter++) {
    if (K2_eval < 1e-20) K2_eval = 1e-20;
    double tnew = t - K1_eval / K2_eval;

    if (std::abs(tnew - t) < tol) {
      res.zeta = tnew;
      res.converged = true;
      return res;
    }

    cgf = compute_binom_cgf(tnew, cache);
    double new_K1 = cgf.K1 - q;

    if (std::isnan(tnew) || std::isnan(new_K1)) break;

    if ((K1_eval > 0) != (new_K1 > 0)) {
      if (std::abs(tnew - t) > (prev_jump - tol)) {
        tnew = t + ((new_K1 > K1_eval) ? 1.0 : -1.0) * prev_jump / 2.0;
        cgf = compute_binom_cgf(tnew, cache);
        new_K1 = cgf.K1 - q;
        prev_jump = prev_jump / 2.0;
      } else {
        prev_jump = std::abs(tnew - t);
      }
    }

    t = tnew;
    K1_eval = new_K1;
    K2_eval = cgf.K2;
  }

  res.zeta = t;
  res.converged = false;
  return res;
}

double spa_pvalue_one_tail_binom(double zeta, double q,
                                 const std::vector<BinomCgfObsCache> &cache) {
  auto cgf = compute_binom_cgf(zeta, cache);
  double temp1 = zeta * q - cgf.K;
  if (!std::isfinite(cgf.K) || !std::isfinite(cgf.K2) || temp1 < 0 ||
      cgf.K2 < 0) {
    return -1.0;
  }
  double w = (zeta > 0 ? 1.0 : -1.0) * std::sqrt(2.0 * temp1);
  double v = zeta * std::sqrt(cgf.K2);
  if (std::abs(w) < 1e-15) return -1.0;
  double z_spa = w + (1.0 / w) * std::log(v / w);
  if (std::isnan(z_spa)) return -1.0;
  if (z_spa > 0) {
    return R::pnorm(z_spa, 0.0, 1.0, 0, 0);
  } else {
    return R::pnorm(z_spa, 0.0, 1.0, 1, 0);
  }
}

double spa_pvalue_twosided_binom(double q, double pval_nospa,
                                 const std::vector<BinomCgfObsCache> &cache,
                                 double tol = 1e-8) {
  auto sp1 = find_saddlepoint_binom(q, cache, tol);
  auto sp2 = find_saddlepoint_binom(-q, cache, tol);
  if (!sp1.converged || !sp2.converged) return pval_nospa;
  double p1 = spa_pvalue_one_tail_binom(sp1.zeta, q, cache);
  double p2 = spa_pvalue_one_tail_binom(sp2.zeta, -q, cache);
  if (p1 < 0 || p2 < 0) return pval_nospa;
  return std::abs(p1) + std::abs(p2);
}

// [[Rcpp::export]]
Rcpp::List score_test_zero_cpp(const arma::vec &null_par, const arma::vec &Y,
                               const arma::mat &Z_null, const arma::mat &Z_full,
                               const arma::vec &offsetz,
                               const arma::vec &weights, bool use_spa = false,
                               double spa_cutoff = 2.0) {
  int kz_null = Z_null.n_cols;
  int kz_full = Z_full.n_cols;
  int n = Y.n_elem;

  // Require at least one null covariate (intercept)
  if (kz_null < 1) {
    Rcpp::stop("Z_null must have at least one column (intercept)");
  }

  // Binary response
  arma::vec y_bin(n);
  for (int i = 0; i < n; i++) y_bin(i) = (Y(i) > 0) ? 1.0 : 0.0;

  // Null fitted values: eta = Z_null * beta + offset, p = logistic(eta)
  arma::vec eta_null = Z_null * null_par + offsetz;
  arma::vec p_null = 1.0 / (1.0 + arma::exp(-eta_null));

  // Score for the test variable: U = Σ w * (y_bin - p) * z_test
  // Use the full Z matrix to include the test column
  arma::vec residuals = y_bin - p_null;
  arma::vec z_test = Z_full.col(kz_null);  // test column is at position kz_null
  double U_test = arma::dot(weights % residuals, z_test);

  // FIM: Z_full' diag(w * p * (1-p)) Z_full
  arma::vec W_diag = weights % p_null % (1.0 - p_null);
  arma::mat Z_weighted = Z_full.each_col() % W_diag;
  arma::mat fim = Z_full.t() * Z_weighted;

  // Schur complement: project out null covariates
  arma::mat I_nn = fim.submat(0, 0, kz_null - 1, kz_null - 1);
  arma::mat I_nt = fim.submat(0, kz_null, kz_null - 1, kz_full - 1);
  arma::mat I_tt = fim.submat(kz_null, kz_null, kz_full - 1, kz_full - 1);

  // Guard solve() for singular/ill-conditioned I_nn
  // Computation failures return NA (matching Wald behavior for pipeline
  // robustness)
  arma::mat I_nn_inv_Int;
  bool solve_ok = arma::solve(I_nn_inv_Int, I_nn, I_nt);
  if (!solve_ok) {
    return Rcpp::List::create(
        Rcpp::Named("beta") = Rcpp::NumericVector::create(NA_REAL),
        Rcpp::Named("se") = Rcpp::NumericVector::create(NA_REAL),
        Rcpp::Named("statistic") = NA_REAL, Rcpp::Named("pvalue") = NA_REAL,
        Rcpp::Named("spa_applied") = false);
  }
  arma::mat I_test = I_tt - I_nt.t() * I_nn_inv_Int;

  double I_val = I_test(0, 0);
  if (I_val <= 0 || !std::isfinite(I_val)) {
    return Rcpp::List::create(
        Rcpp::Named("beta") = Rcpp::NumericVector::create(NA_REAL),
        Rcpp::Named("se") = Rcpp::NumericVector::create(NA_REAL),
        Rcpp::Named("statistic") = NA_REAL, Rcpp::Named("pvalue") = NA_REAL,
        Rcpp::Named("spa_applied") = false);
  }

  double T_stat = U_test * U_test / I_val;
  double beta_hat = U_test / I_val;
  double pvalue = R::pchisq(T_stat, 1.0, 0, 0);

  // SPA
  bool spa_applied = false;
  if (use_spa && std::sqrt(T_stat) > spa_cutoff) {
    // Covariate-projected test variable
    arma::vec proj_coef = arma::solve(I_nn, I_nt.col(0));
    arma::vec g_tilde =
        Z_full.col(kz_null) - Z_full.cols(0, kz_null - 1) * proj_coef;

    // Build binomial CGF cache
    std::vector<BinomCgfObsCache> cgf_cache;
    cgf_cache.reserve(n);
    for (int i = 0; i < n; i++) {
      double gi = g_tilde(i);
      if (std::abs(gi) < 1e-15) continue;
      double pi = p_null(i);
      if (pi < 1e-15 || pi > 1.0 - 1e-15) continue;
      BinomCgfObsCache obs;
      obs.pi = pi;
      obs.gi = gi;
      obs.wi = weights(i);
      cgf_cache.push_back(obs);
    }

    // U_test = Σ w*(y-p)*g_tilde is already the centered score.
    // The CGF is mean-centered (K'(0)=0), so pass U_test directly.
    if (!cgf_cache.empty()) {
      double p_spa = spa_pvalue_twosided_binom(U_test, pvalue, cgf_cache);
      if (p_spa >= 0 && p_spa <= 1.0) {
        pvalue = p_spa;
        spa_applied = true;
      }
    }
  }

  // Beta refinement: 5-iter BFGS for significant tests
  if (spa_applied) {
    arma::vec start_refine(kz_full);
    start_refine.subvec(0, kz_null - 1) = null_par;
    start_refine(kz_null) = beta_hat;

    ZeroBinomFunctor refine_functor(Y, Z_full, offsetz, weights, "logit");
    double obj_before = refine_functor(start_refine);

    arma::vec par = start_refine;
    Roptim<ZeroBinomFunctor> opt("BFGS");
    opt.control.trace = 0;
    opt.control.maxit = 5;
    opt.set_hessian(false);
    opt.minimize(refine_functor, par);

    double obj_after = opt.value();
    double beta_refined = opt.par()(kz_null);
    if (std::isfinite(beta_refined) && std::isfinite(obj_after) &&
        obj_after <= obj_before) {
      beta_hat = beta_refined;
    }
  }

  // SE: back-computed from p-value
  double se_hat;
  if (pvalue > 0.0 && pvalue < 1.0) {
    double z_abs = R::qnorm(pvalue / 2.0, 0.0, 1.0, 0, 0);
    se_hat = std::abs(beta_hat) / z_abs;
  } else {
    se_hat = (pvalue == 0.0) ? 0.0 : R_PosInf;
  }

  return Rcpp::List::create(
      Rcpp::Named("beta") = Rcpp::NumericVector::create(beta_hat),
      Rcpp::Named("se") = Rcpp::NumericVector::create(se_hat),
      Rcpp::Named("statistic") = T_stat, Rcpp::Named("pvalue") = pvalue,
      Rcpp::Named("spa_applied") = spa_applied);
}

// [[Rcpp::export]]
Rcpp::List compute_negbin_hurdle_fitted_cpp(
    const arma::vec &coefc, const arma::vec &coefz, const arma::mat &X,
    const arma::mat &Z, const arma::vec &offsetx, const arma::vec &offsetz,
    double theta, const arma::vec &y) {
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

  return Rcpp::List::create(Rcpp::Named("fitted.values") = Yhat,
                            Rcpp::Named("residuals") = res);
}
