#include <RcppArmadillo.h>

#include <cmath>
#include <limits>
#include <vector>

#include "links.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;
using namespace Rcpp;

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

// Phase 4: Build CGF cache from pre-computed mu/p0/log_p1 (avoids recomputing
// eta/mu/p0/r that were already computed in the fused score+Hessian function).
std::vector<ZtnbCgfObsCache> build_cgf_cache_from_intermediates(
    double theta, const arma::vec &mu_pos, const arma::vec &p0_pos,
    const arma::vec &log_p1_pos, const arma::vec &w_pos,
    const arma::vec &g_tilde_pos) {
  int n_pos = mu_pos.n_elem;
  std::vector<ZtnbCgfObsCache> cache;
  cache.reserve(n_pos);

  for (int i = 0; i < n_pos; i++) {
    double gi = g_tilde_pos(i);
    if (std::abs(gi) < 1e-15) continue;

    double mu = mu_pos(i);
    double p0 = p0_pos(i);
    double log_p1 = log_p1_pos(i);
    double p1 = std::exp(log_p1);
    if (p1 < 1e-300) {
      cache.clear();
      return cache;
    }
    double r = p0 / p1;
    double mu_theta = mu + theta;
    double alpha = theta / mu_theta;

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
// Score test for count component: unified path via prepare_score_cache_count_cpp +
// score_test_count_cpp (defined below in the cached section).
// ==========================================================================

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

// ==========================================================================
// Zero score test: pre-compute null-only quantities ONCE via
// prepare_score_cache_zero_cpp, then run per-peak tests with O(n) dot
// products + O(kz^2) Schur complement.
// ==========================================================================

// [[Rcpp::export]]
Rcpp::List prepare_score_cache_zero_cpp(
    const arma::vec &null_par, const arma::vec &Y,
    const arma::mat &Z_null, const arma::vec &offsetz,
    const arma::vec &weights) {

  int kz_null = Z_null.n_cols;
  int n = Y.n_elem;

  // Binary response
  arma::vec y_bin(n);
  for (int i = 0; i < n; i++) y_bin(i) = (Y(i) > 0) ? 1.0 : 0.0;

  // Null fitted values
  arma::vec eta_null = Z_null * null_par + offsetz;
  arma::vec p_null = 1.0 / (1.0 + arma::exp(-eta_null));

  // Score weights (constant across test variables)
  arma::vec W_resid = weights % (y_bin - p_null);

  // FIM diagonal weights
  arma::vec W_diag = weights % p_null % (1.0 - p_null);

  // Null FIM and its inverse
  arma::mat Z_w = Z_null.each_col() % W_diag;
  arma::mat I_nn = Z_null.t() * Z_w;
  arma::mat I_nn_inv;
  bool inv_ok = arma::inv_sympd(I_nn_inv, I_nn);
  if (!inv_ok) inv_ok = arma::inv(I_nn_inv, I_nn);
  if (!inv_ok) {
    return Rcpp::List::create(Rcpp::Named("valid") = false);
  }

  // Cache Z_null' diag(W) for cross-FIM: kz_null x n
  arma::mat Znull_W_t = Z_w.t();

  // SPA intermediates
  arma::vec p_null_cache = p_null;

  return Rcpp::List::create(
    Rcpp::Named("valid") = true,
    Rcpp::Named("W_resid") = W_resid,
    Rcpp::Named("W_diag") = W_diag,
    Rcpp::Named("I_nn_inv") = I_nn_inv,
    Rcpp::Named("Znull_W_t") = Znull_W_t,
    Rcpp::Named("p_null") = p_null_cache,
    Rcpp::Named("kz_null") = kz_null);
}

// Joint IRLS refinement for logistic (zero component) test variable.
// eta_null = Z*beta + offset (+ random effects for GLMM) — full null linear
// predictor. off = eta_null - Z_null * beta_null absorbs offset + RE so that
// the Schur solve updates (beta_null, beta_test) in fixed-effect space.
// Same 3-step pattern as refine_beta_joint_ztnb.
static double refine_beta_joint_logistic(
    double beta_init, const arma::vec &z_test, const arma::vec &y_bin,
    const arma::vec &eta_null,  // full null eta (Z*beta + offset [+ b_j])
    const arma::mat &Z_null,    // n x kz
    const arma::vec &weights,   // prior weights
    arma::vec &beta_null,       // in/out: null fixed effects
    int max_iter = 3) {

  const int n  = z_test.n_elem;
  const int kz = Z_null.n_cols;
  double beta_test = beta_init;

  arma::vec W_vec(n), z_vec(n), eta(n);
  arma::vec off = eta_null - Z_null * beta_null; // offset + random effects

  for (int iter = 0; iter < max_iter; iter++) {
    eta = Z_null * beta_null + off + beta_test * z_test;

    for (int i = 0; i < n; i++) {
      double mu_i = 1.0 / (1.0 + std::exp(-eta(i)));
      mu_i = std::max(1e-10, std::min(1.0 - 1e-10, mu_i));
      double D_i  = mu_i * (1.0 - mu_i);
      W_vec(i) = D_i * weights(i);
      z_vec(i) = (eta(i) - off(i)) + (y_bin(i) - mu_i) / D_i;
    }

    arma::vec Wz_t      = W_vec % z_test;
    arma::mat WZ        = Z_null.each_col() % W_vec;
    arma::mat ZtWZ      = Z_null.t() * WZ;
    arma::vec ZtWz_t    = Z_null.t() * Wz_t;
    arma::vec ZtWz_resp = Z_null.t() * (W_vec % z_vec);

    arma::mat ZtWZ_inv;
    bool ok = arma::solve(ZtWZ_inv, ZtWZ, arma::eye(kz, kz),
                          arma::solve_opts::likely_sympd);
    if (!ok) ok = arma::inv(ZtWZ_inv, ZtWZ);
    if (!ok) break;

    double zWz_resp      = arma::dot(Wz_t, z_vec);
    double zWz           = arma::dot(Wz_t, z_test);
    double zt_inv_zz_resp = arma::dot(ZtWz_t, ZtWZ_inv * ZtWz_resp);
    double zt_inv_zz      = arma::dot(ZtWz_t, ZtWZ_inv * ZtWz_t);
    double denom = zWz - zt_inv_zz;
    if (denom <= 0 || !std::isfinite(denom)) break;

    double beta_new = (zWz_resp - zt_inv_zz_resp) / denom;
    beta_null = ZtWZ_inv * (ZtWz_resp - ZtWz_t * beta_new);

    if (std::abs(beta_new - beta_test) < 1e-6 * (std::abs(beta_test) + 1e-8)) {
      beta_test = beta_new;
      break;
    }
    beta_test = beta_new;
  }
  return std::isfinite(beta_test) ? beta_test : beta_init;
}

// Per-peak zero score test
// [[Rcpp::export]]
Rcpp::List score_test_zero_cpp(
    const arma::vec &z_test,           // n
    const arma::vec &W_resid,          // n (cached: w * (y - p))
    const arma::vec &W_diag,           // n (cached: w * p * (1-p))
    const arma::mat &I_nn_inv,         // kz x kz
    const arma::mat &Znull_W_t,        // kz x n
    const arma::vec &p_null,           // n (for SPA)
    const arma::vec &Y,                // n (for beta refinement)
    const arma::mat &Z_null,           // n x kz (for SPA projection + refinement)
    const arma::vec &offsetz,          // n
    const arma::vec &weights,          // n
    const arma::vec &null_par,         // kz (for beta refinement)
    int kz_null,
    bool use_spa = false, double spa_cutoff = 2.0) {

  // Score: U = dot(W_resid, z_test)
  double U_test = arma::dot(W_resid, z_test);

  // Cross-FIM: I_nt = Znull_W_t * z_test (kz x 1)
  arma::vec I_nt = Znull_W_t * z_test;

  // Test-test FIM: I_tt = z_test' diag(W) z_test
  double I_tt = arma::dot(W_diag % z_test, z_test);

  // Schur complement
  double I_eff = I_tt - arma::dot(I_nt, I_nn_inv * I_nt);

  if (I_eff <= 0 || !std::isfinite(I_eff)) {
    return Rcpp::List::create(
      Rcpp::Named("beta") = Rcpp::NumericVector::create(NA_REAL),
      Rcpp::Named("se") = Rcpp::NumericVector::create(NA_REAL),
      Rcpp::Named("statistic") = NA_REAL,
      Rcpp::Named("pvalue") = NA_REAL,
      Rcpp::Named("spa_applied") = false);
  }

  double T_stat = U_test * U_test / I_eff;
  double beta_hat = U_test / I_eff;
  double pvalue = R::pchisq(T_stat, 1.0, 0, 0);

  // SPA
  bool spa_applied = false;
  if (use_spa && std::sqrt(T_stat) > spa_cutoff) {
    arma::vec proj_coef = I_nn_inv * I_nt;
    arma::vec g_tilde = z_test - Z_null * proj_coef;
    int n = z_test.n_elem;
    std::vector<BinomCgfObsCache> cgf_cache;
    cgf_cache.reserve(n);
    for (int i = 0; i < n; i++) {
      double gi = g_tilde(i);
      if (std::abs(gi) < 1e-15) continue;
      double pi = p_null(i);
      if (pi < 1e-15 || pi > 1.0 - 1e-15) continue;
      BinomCgfObsCache obs;
      obs.pi = pi; obs.gi = gi; obs.wi = weights(i);
      cgf_cache.push_back(obs);
    }
    if (!cgf_cache.empty()) {
      double p_spa = spa_pvalue_twosided_binom(U_test, pvalue, cgf_cache);
      if (p_spa >= 0 && p_spa <= 1.0) {
        pvalue = p_spa;
        spa_applied = true;
      }
    }
  }

  // Beta refinement: apply joint logistic IRLS when test statistic is
  // significant (|z| > 2). Mirrors the count path's refine_beta_joint_ztnb.
  // SE is always back-computed from p-value below, so only beta changes here.
  if (std::sqrt(T_stat) > 2.0) {
    int nz = z_test.n_elem;
    arma::vec y_bin(nz);
    for (int i = 0; i < nz; i++) y_bin(i) = (Y(i) > 0) ? 1.0 : 0.0;
    arma::vec eta_null(nz);
    for (int i = 0; i < nz; i++) {
      double p = std::max(1e-10, std::min(1.0 - 1e-10, p_null(i)));
      eta_null(i) = std::log(p / (1.0 - p));
    }
    arma::vec bn = null_par;
    double beta_refined = refine_beta_joint_logistic(
      beta_hat, z_test, y_bin, eta_null, Z_null, weights, bn);
    if (std::isfinite(beta_refined)) beta_hat = beta_refined;
  }

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
    Rcpp::Named("statistic") = T_stat,
    Rcpp::Named("pvalue") = pvalue,
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

// ==========================================================================
// Cached score test: pre-compute null-only quantities ONCE, then run
// per-peak tests with only O(n_pos) dot products + O(kx^2) Schur complement.
// ==========================================================================

// [[Rcpp::export]]
Rcpp::List prepare_score_cache_count_cpp(
    const arma::vec &null_par, const arma::vec &Y,
    const arma::mat &X_null, const arma::vec &offsetx,
    const arma::vec &weights,
    const std::string &dist = "negbin") {

  int kx_null = X_null.n_cols;
  bool has_theta = (dist == "negbin");
  double theta;
  arma::vec beta_null;
  if (has_theta) {
    theta = std::exp(null_par(kx_null));
    beta_null = null_par.subvec(0, kx_null - 1);
  } else if (dist == "geometric") {
    theta = 1.0;
    beta_null = null_par;
  } else {
    // poisson: NB with very large theta approximates Poisson
    theta = 1e8;
    beta_null = null_par;
  }

  // Pre-subset Y>0
  arma::uvec Y1 = arma::find(Y > 0);
  if (Y1.n_elem == 0) {
    return Rcpp::List::create(Rcpp::Named("valid") = false);
  }
  arma::vec Y_pos = Y.elem(Y1);
  arma::mat X_null_pos = X_null.rows(Y1);
  arma::vec off_pos = offsetx.elem(Y1);
  arma::vec w_pos = weights.elem(Y1);
  int n_pos = Y1.n_elem;

  // Compute eta and mu at null MLE
  arma::vec eta_null = X_null_pos * beta_null + off_pos;
  arma::vec mu_null = arma::exp(eta_null);

  // Digamma/trigamma lookup table (only needed for negbin theta estimation)
  double log_theta = std::log(theta);
  std::vector<double> digamma_tab, trigamma_tab;
  int tab_max = 0;
  if (has_theta) {
    int raw_max_y = static_cast<int>(Y_pos.max());
    tab_max = std::min(raw_max_y, 10000);
    digamma_tab.resize(tab_max + 1, 0.0);
    trigamma_tab.resize(tab_max + 1, 0.0);
    for (int k = 1; k <= tab_max; k++) {
      double tk = theta + static_cast<double>(k - 1);
      digamma_tab[k] = digamma_tab[k - 1] + 1.0 / tk;
      trigamma_tab[k] = trigamma_tab[k - 1] + 1.0 / (tk * tk);
    }
  }

  // Single pass: compute all null-only quantities
  arma::vec grad_weights(n_pos);  // w * grad_term (for U = dot(grad_weights, x_test_pos))
  arma::vec v_ee(n_pos);          // Hessian beta-beta weight per obs
  arma::vec v_et;                 // Hessian beta-theta weight per obs (negbin only)
  double v_tt_sum = 0.0;
  if (has_theta) v_et.set_size(n_pos);
  arma::vec mu_pos(n_pos), p0_pos(n_pos), log_p1_pos(n_pos);

  for (int i = 0; i < n_pos; i++) {
    double mu = mu_null(i);
    double y = Y_pos(i);
    double A = mu + theta;
    double A2 = A * A;
    double mu_over_A = mu / A;

    double log_p0 = -theta * std::log1p(mu / theta);
    double p0 = std::exp(log_p0);
    double p1 = 1.0 - p0;
    if (p1 < 1e-300) p1 = 1e-300;
    double r = p0 / p1;

    // SPA intermediates
    mu_pos(i) = mu;
    p0_pos(i) = p0;
    log_p1_pos(i) =
        (p0 > 0.5) ? std::log(-std::expm1(log_p0)) : std::log1p(-p0);

    // Score weight: U_test = dot(grad_weights, x_test_pos)
    double grad_term_i = y - mu * (y + theta) / A - r * theta * mu_over_A;
    grad_weights(i) = w_pos(i) * grad_term_i;

    // Hessian weights
    double a_eta = -theta * mu_over_A;
    double a_ee = -theta * theta * mu / A2;

    double I_NB_ee = mu * theta * (y + theta) / A2;
    double ZT_ee = -r * a_ee - r * (1.0 + r) * a_eta * a_eta;
    v_ee(i) = w_pos(i) * (I_NB_ee + ZT_ee);

    if (has_theta) {
      double a_theta = -std::log1p(mu / theta) + mu_over_A;
      double a_et = -mu * mu / A2;
      double a_tt = mu * mu / (theta * A2);

      double I_NB_et = mu * (mu - y) / A2;
      double ZT_et = -r * a_et - r * (1.0 + r) * a_eta * a_theta;
      v_et(i) = w_pos(i) * theta * (I_NB_et + ZT_et);

      // v_tt: needs digamma/trigamma
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
      v_tt_sum += w_pos(i) *
                  (-theta * first_deriv_theta - theta * theta * second_deriv_theta);
    }
  }

  // Assemble null FIM block
  // has_theta: I_nn = [X' diag(v_ee) X, X' v_et; v_et' X, v_tt_sum], (kx_null+1) x (kx_null+1)
  // !has_theta: I_nn = X' diag(v_ee) X, kx_null x kx_null
  int np_null = has_theta ? kx_null + 1 : kx_null;
  arma::mat I_nn(np_null, np_null, arma::fill::zeros);
  arma::mat X_vee = X_null_pos.each_col() % v_ee;
  I_nn.submat(0, 0, kx_null - 1, kx_null - 1) = X_null_pos.t() * X_vee;
  if (has_theta) {
    arma::vec xt_vet = X_null_pos.t() * v_et;
    I_nn.submat(0, kx_null, kx_null - 1, kx_null) = xt_vet;
    I_nn.submat(kx_null, 0, kx_null, kx_null - 1) = xt_vet.t();
    I_nn(kx_null, kx_null) = v_tt_sum;
  }

  // Pre-compute I_nn inverse for per-peak Schur complement.
  // Matrix is small (typically 3-6), so explicit inverse is fine.
  arma::mat I_nn_inv;
  bool inv_ok = arma::inv_sympd(I_nn_inv, I_nn);
  if (!inv_ok) inv_ok = arma::inv(I_nn_inv, I_nn);
  if (!inv_ok) {
    return Rcpp::List::create(Rcpp::Named("valid") = false);
  }

  // Beta-only FIM block inverse for SPA projection
  arma::mat I_nn_beta_inv;
  bool beta_inv_ok;
  if (has_theta) {
    arma::mat I_nn_beta = I_nn.submat(0, 0, kx_null - 1, kx_null - 1);
    beta_inv_ok = arma::inv_sympd(I_nn_beta_inv, I_nn_beta);
    if (!beta_inv_ok) beta_inv_ok = arma::inv(I_nn_beta_inv, I_nn_beta);
  } else {
    // Without theta, I_nn IS the beta-only block
    I_nn_beta_inv = I_nn_inv;
    beta_inv_ok = true;
  }

  // Cache X_null_pos weighted by v_ee for cross-FIM terms
  arma::mat Xnull_vee_t = X_vee.t();  // kx_null x n_pos

  // Pre-compute eta at null MLE for Newton refinement
  arma::vec eta_null_pos = X_null_pos * beta_null + off_pos;

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("valid") = true,
    Rcpp::Named("has_theta") = has_theta,
    Rcpp::Named("Y1") = Y1,
    Rcpp::Named("Y_pos") = Y_pos,
    Rcpp::Named("grad_weights") = grad_weights,
    Rcpp::Named("v_ee") = v_ee,
    Rcpp::Named("I_nn_inv") = I_nn_inv,
    Rcpp::Named("I_nn_beta_inv") = I_nn_beta_inv,
    Rcpp::Named("beta_inv_ok") = beta_inv_ok,
    Rcpp::Named("Xnull_vee_t") = Xnull_vee_t,
    Rcpp::Named("X_null_pos") = X_null_pos,
    Rcpp::Named("w_pos") = w_pos,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("beta_null") = beta_null,
    Rcpp::Named("eta_null_pos") = eta_null_pos,
    Rcpp::Named("mu_pos") = mu_pos,
    Rcpp::Named("p0_pos") = p0_pos,
    Rcpp::Named("log_p1_pos") = log_p1_pos,
    Rcpp::Named("kx_null") = kx_null);
  if (has_theta) {
    result["v_et"] = v_et;
  }
  return result;
}

// Joint IRLS refinement for beta_test via Schur complement.
// Jointly re-estimates beta_null and beta_test using ZTNB IRLS working
// quantities, solving the (kx+1)-dim system via Schur complement so
// that only beta_test is extracted. 3 iterations suffice for <1% error.
// Cost: O(n_pos * kx) per iteration (dominated by ZTNB wq, same as 1D).
static double refine_beta_joint_ztnb(
    double beta_init, const arma::vec &x_pos, const arma::vec &y_pos,
    const arma::vec &eta_null_pos,   // null eta (X*beta_null + offset)
    const arma::mat &X_null_pos,     // n_pos x kx
    const arma::vec &w_pos, double theta,
    arma::vec &beta_null,            // in/out: updated jointly
    int max_iter = 3) {

  const double theta_f = std::max(theta, 0.01);
  const int n_pos = x_pos.n_elem;
  const int kx = X_null_pos.n_cols;
  double beta_test = beta_init;

  // Pre-allocate per-iteration work arrays
  arma::vec W_vec(n_pos), z_vec(n_pos), eta(n_pos);

  // Extract offset: eta_null_pos = X_null * beta_null_orig + offset
  // offset is constant across iterations
  arma::vec off_pos = eta_null_pos - X_null_pos * beta_null;

  for (int iter = 0; iter < max_iter; iter++) {
    eta = X_null_pos * beta_null + off_pos + beta_test * x_pos;

    // ZTNB working quantities (inline, same 3-regime as compute_ztnb_wq)
    for (int i = 0; i < n_pos; i++) {
      double eta_i = eta(i);
      double mu, mu_t, D_val, W_val;

      if (eta_i > 700.0) {
        mu = std::exp(std::min(eta_i, 700.0));
        mu_t = mu; D_val = mu;
        W_val = theta_f * w_pos(i);
      } else {
        mu = std::exp(eta_i);
        double A = mu + theta_f;
        double log_p0 = theta_f * std::log(theta_f / A);
        if (log_p0 < -30.0) {
          mu_t = mu; D_val = mu;
          W_val = theta_f * mu / A * w_pos(i);
        } else {
          double p0 = std::exp(log_p0);
          double p1 = -std::expm1(log_p0);
          if (p1 < 1e-300) p1 = 1e-300;
          mu_t = mu / p1;
          double dp0_dmu = -p0 * theta_f / A;
          double dmu_t_dmu = (p1 + mu * dp0_dmu) / (p1 * p1);
          D_val = dmu_t_dmu * mu;
          double dnb1 = p0 * theta_f * mu / A;
          double mu_star = mu + A * dnb1 / (theta_f * p1);
          double V = mu_star + mu_star * mu * (1.0 + 1.0 / theta_f) - mu_star * mu_star;
          if (V < 1e-300) V = 1e-300;
          W_val = D_val * D_val / V * w_pos(i);
        }
      }
      W_vec(i) = W_val;
      // Working response: z = eta_no_offset + (y - mu_t) / D
      // where eta_no_offset = X*beta + beta_test*x (without offset)
      z_vec(i) = (eta(i) - off_pos(i)) + (y_pos(i) - mu_t) / D_val;
    }

    // Schur complement solve for beta_test (profiling out beta_null)
    arma::vec Wx = W_vec % x_pos;
    arma::mat WX = X_null_pos.each_col() % W_vec;
    arma::mat XtWX = X_null_pos.t() * WX;           // kx x kx
    arma::vec XtWx = X_null_pos.t() * Wx;            // kx x 1
    arma::vec XtWz = X_null_pos.t() * (W_vec % z_vec);

    arma::mat XtWX_inv;
    bool ok = arma::solve(XtWX_inv, XtWX, arma::eye(kx, kx),
                          arma::solve_opts::likely_sympd);
    if (!ok) ok = arma::inv(XtWX_inv, XtWX);
    if (!ok) break;

    double xWz = arma::dot(Wx, z_vec);
    double xWx = arma::dot(Wx, x_pos);
    arma::vec XtWX_inv_XtWx = XtWX_inv * XtWx;
    double xt_inv_xz = arma::dot(XtWx, XtWX_inv * XtWz);
    double xt_inv_xx = arma::dot(XtWx, XtWX_inv_XtWx);

    double denom = xWx - xt_inv_xx;
    if (denom <= 0 || !std::isfinite(denom)) break;

    double beta_test_new = (xWz - xt_inv_xz) / denom;
    beta_null = XtWX_inv * (XtWz - XtWx * beta_test_new);

    if (std::abs(beta_test_new - beta_test) < 1e-6 * (std::abs(beta_test) + 1e-8)) {
      beta_test = beta_test_new;
      break;
    }
    beta_test = beta_test_new;
  }
  return beta_test;
}

// Per-peak score test: only test-variable-dependent work.
// Unified path for all distributions (negbin, poisson, geometric).
// Cache must be prepared via prepare_score_cache_count_cpp first.
// [[Rcpp::export]]
Rcpp::List score_test_count_cpp(
    const arma::vec &x_test,          // length n (full)
    const arma::uvec &Y1,             // pos indices (0-based)
    const arma::vec &grad_weights,    // n_pos
    const arma::vec &v_ee,            // n_pos
    const arma::vec &Y_pos,           // positive counts (for IRLS refinement)
    const arma::mat &I_nn_inv,        // np_null x np_null inverse
    const arma::mat &I_nn_beta_inv,   // beta-only FIM block inverse (for SPA)
    bool beta_inv_ok,
    const arma::mat &Xnull_vee_t,     // kx_null x n_pos
    const arma::mat &X_null_pos,      // n_pos x kx_null
    const arma::vec &w_pos,           // n_pos
    double theta,
    const arma::vec &beta_null,       // kx_null (for joint refinement)
    const arma::vec &eta_null_pos,    // n_pos (pre-computed null eta)
    const arma::vec &mu_pos,          // n_pos (for SPA)
    const arma::vec &p0_pos,          // n_pos (for SPA)
    const arma::vec &log_p1_pos,      // n_pos (for SPA)
    int kx_null,
    bool has_theta = true,            // true for negbin (FIM includes theta), false for poisson/geometric
    bool use_spa = false, double spa_cutoff = 2.0,
    Rcpp::Nullable<arma::vec> v_et_nullable = R_NilValue) {

  arma::vec x_pos = x_test.elem(Y1);

  // 1. Score: single O(n_pos) dot product
  double U_test = arma::dot(grad_weights, x_pos);

  // 2. Cross-FIM terms and Schur complement
  arma::vec I_nt_beta = Xnull_vee_t * x_pos;
  double I_tt = arma::dot(v_ee % x_pos, x_pos);
  double I_eff;

  if (has_theta) {
    // Negbin: I_nn is (kx_null+1) x (kx_null+1), includes theta cross-terms
    arma::vec v_et = Rcpp::as<arma::vec>(v_et_nullable);
    double I_nt_theta = arma::dot(v_et, x_pos);
    arma::vec I_nt(kx_null + 1);
    I_nt.subvec(0, kx_null - 1) = I_nt_beta;
    I_nt(kx_null) = I_nt_theta;
    I_eff = I_tt - arma::dot(I_nt, I_nn_inv * I_nt);
  } else {
    // Poisson/Geometric: I_nn is kx_null x kx_null, beta-only
    I_eff = I_tt - arma::dot(I_nt_beta, I_nn_inv * I_nt_beta);
  }

  if (I_eff <= 0 || !std::isfinite(I_eff)) {
    return Rcpp::List::create(
      Rcpp::Named("beta") = arma::vec(1, arma::fill::value(NA_REAL)),
      Rcpp::Named("se") = arma::vec(1, arma::fill::value(NA_REAL)),
      Rcpp::Named("statistic") = NA_REAL,
      Rcpp::Named("pvalue") = NA_REAL,
      Rcpp::Named("spa_applied") = false);
  }

  double T_stat = U_test * U_test / I_eff;
  double beta_hat = U_test / I_eff;
  double se_hat = 1.0 / std::sqrt(I_eff);
  double pvalue = R::pchisq(T_stat, 1.0, 0, 0);

  // SPA: use beta-only FIM block for projection (matches standard path)
  bool spa_applied = false;
  if (use_spa && std::sqrt(T_stat) > spa_cutoff && beta_inv_ok) {
    arma::vec proj_coef = I_nn_beta_inv * I_nt_beta;
    arma::vec g_tilde_pos = x_pos - X_null_pos * proj_coef;
    auto cgf_cache = build_cgf_cache_from_intermediates(
        theta, mu_pos, p0_pos, log_p1_pos, w_pos, g_tilde_pos);
    if (!cgf_cache.empty()) {
      double p_spa = spa_pvalue_twosided(U_test, pvalue, theta, cgf_cache);
      if (p_spa >= 0 && p_spa <= 1.0) {
        pvalue = p_spa;
        spa_applied = true;
      }
    }
  }

  // Joint IRLS refinement: 3 iterations jointly re-estimating beta_null + beta_test
  // via Schur complement. O(n_pos * kx) per iter, matches full MLE within ~1%.
  double refine_cutoff = use_spa ? spa_cutoff : 2.0;
  if (std::sqrt(T_stat) > refine_cutoff) {
    arma::vec bn = beta_null;  // mutable copy for joint update
    beta_hat = refine_beta_joint_ztnb(beta_hat, x_pos, Y_pos,
                                       eta_null_pos, X_null_pos,
                                       w_pos, theta, bn);
  }

  // Back-compute SE from p-value
  if (pvalue <= 0.0) {
    se_hat = 0.0;
  } else if (pvalue >= 1.0) {
    se_hat = R_PosInf;
  } else {
    double z_val = R::qnorm(pvalue / 2.0, 0.0, 1.0, 1, 0);
    if (std::isfinite(z_val) && z_val != 0.0 && beta_hat != 0.0) {
      se_hat = std::abs(beta_hat / z_val);
    }
  }

  arma::vec beta_vec(1);
  beta_vec(0) = beta_hat;
  arma::vec se_vec(1);
  se_vec(0) = se_hat;
  return Rcpp::List::create(
    Rcpp::Named("beta") = beta_vec,
    Rcpp::Named("se") = se_vec,
    Rcpp::Named("statistic") = T_stat,
    Rcpp::Named("pvalue") = pvalue,
    Rcpp::Named("spa_applied") = spa_applied);
}

// ==========================================================================
// Batch score test for count component (standard, non-GLMM)
// Tests n_peaks variables in a single .Call.
// ==========================================================================

// [[Rcpp::export]]
Rcpp::List score_test_count_batch_cpp(
    const arma::mat &X_test_pos,
    const arma::uvec &Y1,
    const arma::vec &grad_weights,
    const arma::vec &v_ee,
    const arma::vec &Y_pos,
    const arma::mat &I_nn_inv,
    const arma::mat &I_nn_beta_inv,
    bool beta_inv_ok,
    const arma::mat &Xnull_vee_t,
    const arma::mat &X_null_pos,
    const arma::vec &w_pos,
    double theta,
    const arma::vec &beta_null,
    const arma::vec &eta_null_pos,
    const arma::vec &mu_pos,
    const arma::vec &p0_pos,
    const arma::vec &log_p1_pos,
    int kx_null,
    bool has_theta,
    bool use_spa, double spa_cutoff,
    Rcpp::Nullable<arma::vec> v_et_nullable) {

  const int n_peaks = X_test_pos.n_cols;

  // 1. Vectorized scores: U = X_test_pos' * grad_weights (BLAS dgemv)
  arma::vec U_vec = X_test_pos.t() * grad_weights;

  // 2. Vectorized variances via Schur complement
  // I_tt_k = x_k' diag(v_ee) x_k  (test-test info)
  arma::mat V_Xtest = X_test_pos.each_col() % v_ee;
  arma::vec I_tt_vec = arma::sum(X_test_pos % V_Xtest, 0).t();

  // Cross-FIM: I_nt_beta = Xnull_vee_t * x_k for each k → kx_null x n_peaks
  arma::mat I_nt_beta_mat = Xnull_vee_t * X_test_pos;

  arma::vec I_eff_vec(n_peaks);
  if (has_theta) {
    arma::vec v_et = Rcpp::as<arma::vec>(v_et_nullable);
    arma::vec I_nt_theta_vec = X_test_pos.t() * v_et;  // n_peaks
    for (int k = 0; k < n_peaks; k++) {
      arma::vec I_nt(kx_null + 1);
      I_nt.subvec(0, kx_null - 1) = I_nt_beta_mat.col(k);
      I_nt(kx_null) = I_nt_theta_vec(k);
      I_eff_vec(k) = I_tt_vec(k) - arma::dot(I_nt, I_nn_inv * I_nt);
    }
  } else {
    // No theta: I_nn is kx_null x kx_null
    arma::mat Q = I_nn_inv * I_nt_beta_mat;
    arma::vec quad = arma::sum(I_nt_beta_mat % Q, 0).t();
    I_eff_vec = I_tt_vec - quad;
  }

  // 3. Test statistics, p-values, refinement
  arma::vec pval_vec(n_peaks);
  arma::vec beta_hat_vec(n_peaks);
  arma::vec se_vec(n_peaks);
  arma::vec stat_vec(n_peaks);
  arma::vec spa_vec(n_peaks, arma::fill::zeros);

  double refine_cutoff = use_spa ? spa_cutoff : 2.0;

  for (int k = 0; k < n_peaks; k++) {
    double I_eff = I_eff_vec(k);
    if (I_eff <= 0 || !std::isfinite(I_eff)) {
      pval_vec(k) = NA_REAL;
      stat_vec(k) = NA_REAL;
      beta_hat_vec(k) = NA_REAL;
      se_vec(k) = NA_REAL;
      continue;
    }

    double T_stat = U_vec(k) * U_vec(k) / I_eff;
    double beta_hat = U_vec(k) / I_eff;
    double se_hat = 1.0 / std::sqrt(I_eff);
    double pvalue = R::pchisq(T_stat, 1.0, 0, 0);

    // SPA (per-peak, only for significant)
    bool spa_applied = false;
    if (use_spa && std::sqrt(T_stat) > spa_cutoff && beta_inv_ok) {
      arma::vec x_k = X_test_pos.col(k);
      arma::vec proj_coef = I_nn_beta_inv * I_nt_beta_mat.col(k);
      arma::vec g_tilde_pos = x_k - X_null_pos * proj_coef;
      auto cgf_cache = build_cgf_cache_from_intermediates(
          theta, mu_pos, p0_pos, log_p1_pos, w_pos, g_tilde_pos);
      if (!cgf_cache.empty()) {
        double p_spa = spa_pvalue_twosided(U_vec(k), pvalue, theta, cgf_cache);
        if (p_spa >= 0 && p_spa <= 1.0) {
          pvalue = p_spa;
          spa_applied = true;
        }
      }
    }

    // Joint IRLS refinement for significant peaks
    if (std::sqrt(T_stat) > refine_cutoff) {
      arma::vec bn = beta_null;
      beta_hat = refine_beta_joint_ztnb(
        beta_hat, X_test_pos.col(k), Y_pos,
        eta_null_pos, X_null_pos, w_pos, theta, bn);
    }

    // Back-compute SE from p-value
    if (pvalue <= 0.0) {
      se_hat = 0.0;
    } else if (pvalue >= 1.0) {
      se_hat = R_PosInf;
    } else {
      double z_val = R::qnorm(pvalue / 2.0, 0.0, 1.0, 1, 0);
      if (std::isfinite(z_val) && z_val != 0.0 && beta_hat != 0.0) {
        se_hat = std::abs(beta_hat / z_val);
      }
    }

    pval_vec(k) = pvalue;
    stat_vec(k) = T_stat;
    beta_hat_vec(k) = beta_hat;
    se_vec(k) = se_hat;
    spa_vec(k) = spa_applied ? 1.0 : 0.0;
  }

  return Rcpp::List::create(
    Rcpp::Named("pvalue") = pval_vec,
    Rcpp::Named("statistic") = stat_vec,
    Rcpp::Named("beta") = beta_hat_vec,
    Rcpp::Named("se") = se_vec,
    Rcpp::Named("spa_applied") = spa_vec);
}
