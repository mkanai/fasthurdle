#ifndef FASTHURDLE_LINKS_H_
#define FASTHURDLE_LINKS_H_

#include <RcppArmadillo.h>

#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>

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

// Tag type for positive-only (pre-subsetted) data constructor
struct PositiveOnlyTag {};

#endif  // FASTHURDLE_LINKS_H_
