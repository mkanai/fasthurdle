#include <RcppArmadillo.h>
#include <roptim.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include "links.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(roptim)]]

using namespace arma;
using namespace Rcpp;
using namespace roptim;

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
  // Standard constructor: finds Y>0, copies subsets
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

  // Positive-only constructor: caller guarantees all Y > 0.
  // Creates zero-copy aliases for X_pos/Y_pos/etc using Armadillo's
  // auxiliary memory mode, avoiding the O(n_pos * kx) copy.
  LikelihoodFunctor(arma::vec &y_pos, arma::mat &x_pos, arma::vec &offs_pos,
                    arma::vec &w_pos_in, PositiveOnlyTag)
      : Y(y_pos),
        X(x_pos),
        offset(offs_pos),
        weights(w_pos_in),
        Y1(arma::regspace<arma::uvec>(0, y_pos.n_elem - 1)),
        X_pos(x_pos.memptr(), x_pos.n_rows, x_pos.n_cols, false, true),
        offset_pos(offs_pos.memptr(), offs_pos.n_elem, false, true),
        w_pos(w_pos_in.memptr(), w_pos_in.n_elem, false, true),
        Y_pos(y_pos.memptr(), y_pos.n_elem, false, true) {
    // Y0 and X_zero remain empty (no zero observations)
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
  void build_lgamma_cache() {
    if (Y1.n_elem > 0) {
      lgamma_y_1_cached.set_size(Y1.n_elem);
      for (size_t i = 0; i < Y1.n_elem; i++) {
        lgamma_y_1_cached(i) = lgamma(Y_pos(i) + 1.0);
      }
    }
  }

  CountPoissonFunctor(const arma::vec &y, const arma::mat &x,
                      const arma::vec &offs, const arma::vec &w)
      : LikelihoodFunctor(y, x, offs, w) {
    build_lgamma_cache();
  }

  // Positive-only constructor: zero-copy alias
  CountPoissonFunctor(arma::vec &y_pos, arma::mat &x_pos, arma::vec &offs_pos,
                      arma::vec &w_pos_in, PositiveOnlyTag tag)
      : LikelihoodFunctor(y_pos, x_pos, offs_pos, w_pos_in, tag) {
    build_lgamma_cache();
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

  void build_y_caches() {
    if (Y1.n_elem > 0) {
      unique_y_vals = arma::unique(Y_pos);
      y_index_map.set_size(Y_pos.n_elem);
      for (size_t i = 0; i < Y_pos.n_elem; i++) {
        auto it = std::lower_bound(unique_y_vals.begin(), unique_y_vals.end(),
                                   Y_pos(i));
        y_index_map(i) = static_cast<arma::uword>(it - unique_y_vals.begin());
      }
      arma::vec lgamma_unique(unique_y_vals.n_elem);
      for (size_t j = 0; j < unique_y_vals.n_elem; j++) {
        lgamma_unique(j) = lgamma(unique_y_vals(j) + 1.0);
      }
      lgamma_y_1_cached = lgamma_unique.elem(y_index_map);
    }
  }

 public:
  CountNegBinFunctor(const arma::vec &y, const arma::mat &x,
                     const arma::vec &offs, const arma::vec &w)
      : LikelihoodFunctor(y, x, offs, w),
        cached_theta(0.0),
        cache_valid(false) {
    build_y_caches();
  }

  // Positive-only constructor: zero-copy alias of pre-subsetted Y>0 data.
  CountNegBinFunctor(arma::vec &y_pos, arma::mat &x_pos, arma::vec &offs_pos,
                     arma::vec &w_pos_in, PositiveOnlyTag tag)
      : LikelihoodFunctor(y_pos, x_pos, offs_pos, w_pos_in, tag),
        cached_theta(0.0),
        cache_valid(false) {
    build_y_caches();
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

  // Phase 2: Vectorize eta/mu computation via BLAS DGEMV instead of per-obs
  // dot products. This is 5-10x faster due to memory locality and SIMD.
  arma::vec eta_vec = X_pos * beta + offset_pos;
  arma::vec mu_vec = arma::exp(eta_vec);

  // Pre-compute digamma/trigamma recurrence lookup table.
  // Capped at 10000 to avoid memory issues with extreme outlier counts;
  // observations with y > tab_max fall back to inline recurrence.
  int raw_max_y = static_cast<int>(Y_pos.max());
  int tab_max = std::min(raw_max_y, 10000);
  std::vector<double> digamma_tab(tab_max + 1, 0.0);
  std::vector<double> trigamma_tab(tab_max + 1, 0.0);
  for (int k = 1; k <= tab_max; k++) {
    double tk = theta + static_cast<double>(k - 1);
    digamma_tab[k] = digamma_tab[k - 1] + 1.0 / tk;
    trigamma_tab[k] = trigamma_tab[k - 1] + 1.0 / (tk * tk);
  }

  for (int i = 0; i < n_pos; i++) {
    double mu = mu_vec(i);
    double y = Y_pos(i);
    double A = mu + theta;
    double A2 = A * A;
    double mu_over_A = mu / A;

    // Zero-truncation quantities
    double log_p0 = -theta * std::log1p(mu / theta);
    double p0 = std::exp(log_p0);
    double p1 = 1.0 - p0;
    if (p1 < 1e-300) p1 = 1e-300;
    double r = p0 / p1;

    // Zero-truncation first derivatives of log(p0)
    double a_eta = -theta * mu_over_A;
    double a_theta = -std::log1p(mu / theta) + mu_over_A;

    // Zero-truncation second derivatives of log(p0)
    double a_ee = -theta * theta * mu / A2;
    double a_et = -mu * mu / A2;
    double a_tt = mu * mu / (theta * A2);

    // I_obs = I_NB + d²log(1-p0)/dudv
    //       = I_NB - r*a_uv - r*(1+r)*a_u*a_v

    // ---- Beta-beta weight (w_ee) ----
    double I_NB_ee = mu * theta * (y + theta) / A2;
    double ZT_ee = -r * a_ee - r * (1.0 + r) * a_eta * a_eta;
    result.v_ee(i) = w_pos(i) * (I_NB_ee + ZT_ee);

    // ---- Beta-logtheta weight (w_et) ----
    double I_NB_et = mu * (mu - y) / A2;
    double ZT_et = -r * a_et - r * (1.0 + r) * a_eta * a_theta;
    result.v_et(i) = w_pos(i) * theta * (I_NB_et + ZT_et);

    // ---- Logtheta-logtheta weight (w_tt) ----
    // Lookup table with inline recurrence fallback for extreme counts
    int yi = static_cast<int>(y);
    double digamma_diff, trigamma_diff;
    if (yi <= tab_max) {
      digamma_diff = digamma_tab[yi];
      trigamma_diff = trigamma_tab[yi];
    } else {
      digamma_diff = 0.0;
      trigamma_diff = 0.0;
      for (int k = 0; k < yi; k++) {
        double tk = theta + static_cast<double>(k);
        digamma_diff += 1.0 / tk;
        trigamma_diff += 1.0 / (tk * tk);
      }
    }

    double log_ratio = log_theta - std::log(A);
    double b = digamma_diff + log_ratio + 1.0 - (y + theta) / A;
    double c = -trigamma_diff + 1.0 / theta - 1.0 / A + (y - mu) / A2;

    double I_ZT_tt = r * a_tt + r * (1.0 + r) * a_theta * a_theta;

    // I_obs[tau,tau] = -theta*(b + r*a_theta) - theta²*(c + I_ZT_tt)
    double first_deriv_theta = b + r * a_theta;
    double second_deriv_theta = c + I_ZT_tt;
    double w_tt_i =
        -theta * first_deriv_theta - theta * theta * second_deriv_theta;

    result.v_tt_sum += w_pos(i) * w_tt_i;
  }

  return result;
}

// Convenience wrapper: compute analytical observed Hessian, assembled.
arma::mat compute_ztnb_observed_info_analytical(const arma::vec &beta,
                                                double theta,
                                                const arma::mat &X_pos,
                                                const arma::vec &offset_pos,
                                                const arma::vec &w_pos,
                                                const arma::vec &Y_pos) {
  auto comp = compute_ztnb_observed_hessian_components(
      beta, theta, X_pos, offset_pos, w_pos, Y_pos);
  return assemble_fim(comp, X_pos);
}

// ==========================================================================
// Fused score + observed Hessian for NB count score test
// ==========================================================================

// Computes BOTH the score statistic U_test and the observed Hessian components
// in a single pass over the data. This eliminates:
// 1. Duplicate eta/mu computation (gradient + Hessian computed them separately)
// 2. The full kx-vector DGEMV for the gradient (only need scalar dot for test
// var)
// 3. Separate log_p0/p0/r computation in the gradient
// Returns pre-computed mu/p0/log_p1 for optional SPA cache reuse (Phase 4).
struct ScoreAndHessianResult {
  double U_test;         // Score for test variable (d logL / d beta_test)
  ZtnbFimResult hess;    // Hessian components (v_ee, v_et, v_tt_sum)
  arma::vec mu_pos;      // Pre-computed mu per obs (for SPA cache)
  arma::vec p0_pos;      // Pre-computed p0 per obs
  arma::vec log_p1_pos;  // Pre-computed log(1-p0) per obs
};

ScoreAndHessianResult compute_score_and_hessian_nb(
    const arma::vec &beta_full, double theta, const arma::mat &X_pos,
    const arma::vec &off_pos, const arma::vec &w_pos, const arma::vec &Y_pos,
    int kx_null, bool store_spa_intermediates = false) {
  int n_pos = X_pos.n_rows;
  double log_theta = std::log(theta);

  ScoreAndHessianResult result;
  result.hess.v_ee.zeros(n_pos);
  result.hess.v_et.zeros(n_pos);
  result.hess.v_tt_sum = 0.0;

  // Vectorize eta/mu via BLAS DGEMV
  arma::vec eta_vec = X_pos * beta_full + off_pos;
  arma::vec mu_vec = arma::exp(eta_vec);

  // Pre-compute digamma/trigamma recurrence lookup table (capped at 10000)
  int raw_max_y = static_cast<int>(Y_pos.max());
  int tab_max = std::min(raw_max_y, 10000);
  std::vector<double> digamma_tab(tab_max + 1, 0.0);
  std::vector<double> trigamma_tab(tab_max + 1, 0.0);
  for (int k = 1; k <= tab_max; k++) {
    double tk = theta + static_cast<double>(k - 1);
    digamma_tab[k] = digamma_tab[k - 1] + 1.0 / tk;
    trigamma_tab[k] = trigamma_tab[k - 1] + 1.0 / (tk * tk);
  }

  // Conditionally allocate SPA reuse buffers only when needed
  if (store_spa_intermediates) {
    result.mu_pos.set_size(n_pos);
    result.p0_pos.set_size(n_pos);
    result.log_p1_pos.set_size(n_pos);
  }

  // Single pass: compute score for test variable AND all Hessian weights
  double U_test_acc = 0.0;

  for (int i = 0; i < n_pos; i++) {
    double mu = mu_vec(i);
    double y = Y_pos(i);
    double A = mu + theta;
    double A2 = A * A;
    double mu_over_A = mu / A;

    // Zero-truncation quantities (shared by score and Hessian)
    double log_p0 = -theta * std::log1p(mu / theta);
    double p0 = std::exp(log_p0);
    double p1 = 1.0 - p0;
    if (p1 < 1e-300) p1 = 1e-300;
    double r = p0 / p1;

    // Store intermediates for SPA cache reuse (only when SPA is enabled)
    if (store_spa_intermediates) {
      result.mu_pos(i) = mu;
      result.p0_pos(i) = p0;
      // log_p1 in log-space for SPA numerical stability
      result.log_p1_pos(i) =
          (p0 > 0.5) ? std::log(-std::expm1(log_p0)) : std::log1p(-p0);
    }

    // ---- Score for test variable ----
    // grad_term = d(logL_ZTNB)/d(eta) = Y - mu*(Y+theta)/A - r*theta*mu/A
    // U_test = Σ w * grad_term * x_test (scalar dot, not full DGEMV)
    double grad_term_i = y - mu * (y + theta) / A - r * theta * mu_over_A;
    U_test_acc += w_pos(i) * grad_term_i * X_pos(i, kx_null);

    // ---- Hessian: beta-beta weight (v_ee) ----
    double a_eta = -theta * mu_over_A;
    double a_theta = -std::log1p(mu / theta) + mu_over_A;
    double a_ee = -theta * theta * mu / A2;
    double a_et = -mu * mu / A2;
    double a_tt = mu * mu / (theta * A2);

    double I_NB_ee = mu * theta * (y + theta) / A2;
    double ZT_ee = -r * a_ee - r * (1.0 + r) * a_eta * a_eta;
    result.hess.v_ee(i) = w_pos(i) * (I_NB_ee + ZT_ee);

    // ---- Hessian: beta-logtheta weight (v_et) ----
    double I_NB_et = mu * (mu - y) / A2;
    double ZT_et = -r * a_et - r * (1.0 + r) * a_eta * a_theta;
    result.hess.v_et(i) = w_pos(i) * theta * (I_NB_et + ZT_et);

    // ---- Hessian: logtheta-logtheta weight (v_tt) ----
    int yi = static_cast<int>(y);
    double digamma_diff, trigamma_diff;
    if (yi <= tab_max) {
      digamma_diff = digamma_tab[yi];
      trigamma_diff = trigamma_tab[yi];
    } else {
      digamma_diff = 0.0;
      trigamma_diff = 0.0;
      for (int k = 0; k < yi; k++) {
        double tk = theta + static_cast<double>(k);
        digamma_diff += 1.0 / tk;
        trigamma_diff += 1.0 / (tk * tk);
      }
    }

    double log_ratio = log_theta - std::log(A);
    double b = digamma_diff + log_ratio + 1.0 - (y + theta) / A;
    double c = -trigamma_diff + 1.0 / theta - 1.0 / A + (y - mu) / A2;

    double I_ZT_tt = r * a_tt + r * (1.0 + r) * a_theta * a_theta;
    double first_deriv_theta = b + r * a_theta;
    double second_deriv_theta = c + I_ZT_tt;
    double w_tt_i =
        -theta * first_deriv_theta - theta * theta * second_deriv_theta;
    result.hess.v_tt_sum += w_pos(i) * w_tt_i;
  }

  result.U_test = U_test_acc;
  return result;
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

  // Positive-only constructor: zero-copy alias
  CountGeomFunctor(arma::vec &y_pos, arma::mat &x_pos, arma::vec &offs_pos,
                   arma::vec &w_pos_in, PositiveOnlyTag tag)
      : LikelihoodFunctor(y_pos, x_pos, offs_pos, w_pos_in, tag),
        negbin_functor(std::make_shared<CountNegBinFunctor>(
            y_pos, x_pos, offs_pos, w_pos_in, tag)) {}

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
