#pragma once

#include <mujoco_rl_training/PolicyIO.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mujoco_rl_training {

// =============================================================================
// RunningMeanStd — Welford streaming mean/variance for observation
// normalization.
//
// Why normalize observations?
//   Different observation dims often have very different scales (e.g. cos/sin
//   angles in [-1, 1] vs angular velocities in [-10, +10] rad/s). MLPs train
//   much faster and more stably when their inputs are roughly zero-mean and
//   unit-std, because all weight initialization schemes assume that scale.
//
// Welford's algorithm (one-pass, numerically stable):
//   Naively computing var = E[x^2] - E[x]^2 catastrophically cancels when
//   E[x^2] ~ E[x]^2. Welford instead updates mean and an accumulator M2
//   (the running sum of squared deviations from the *current* mean):
//
//       count' = count + 1
//       delta  = x - mean
//       mean'  = mean + delta / count'
//       M2'    = M2 + delta * (x - mean')         <-- uses BOTH old and new mean
//       var    = M2 / (count - 1)                 (sample variance, on read)
//
//   This avoids large-cancellation error and works for unbounded streams.
//
// We store M2 (the raw sum) rather than var so updates compose cleanly. On
// read we divide by (count - 1) to get sample variance, taking sqrt and
// adding a tiny epsilon to avoid divide-by-zero in `normalize`.
// =============================================================================
class RunningMeanStd {
   public:
    explicit RunningMeanStd(std::size_t dim) : dim_(dim), mean_(dim, 0.0), m2_(dim, 0.0), count_(0) {}

    std::size_t dim() const { return dim_; }
    std::uint64_t count() const { return count_; }
    const std::vector<double>& mean() const { return mean_; }

    // Per-dim standard deviation. With count <= 1 we have no variance info, so
    // return 1.0 (i.e. pass-through). Caller adds `epsilon` to avoid divide-
    // by-zero from a degenerate constant input dimension.
    std::vector<double> std(double epsilon = 1e-8) const {
        std::vector<double> result(dim_);
        const double divisor = (count_ > 1) ? static_cast<double>(count_ - 1) : 1.0;
        for (std::size_t i = 0; i < dim_; ++i) {
            const double variance = (count_ > 1) ? m2_[i] / divisor : 1.0;
            result[i] = std::sqrt(variance) + epsilon;
        }
        return result;
    }

    // Fold one observation into the running statistics (Welford update).
    void update(const std::vector<double>& obs) {
        if (obs.size() != dim_) {
            throw std::runtime_error("RunningMeanStd::update: obs size mismatch");
        }
        ++count_;
        const double count_d = static_cast<double>(count_);
        for (std::size_t i = 0; i < dim_; ++i) {
            const double delta = obs[i] - mean_[i];
            // Update mean first, then use the updated mean to extend M2.
            mean_[i] += delta / count_d;
            m2_[i] += delta * (obs[i] - mean_[i]);
        }
    }

    // Return (obs - mean) / std, clipped to [-clip, clip] per dim.
    //
    // Clipping is standard practice (Stable Baselines, CleanRL): very early in
    // training, the running mean/std isn't well-estimated and a single
    // out-of-distribution observation can blow up to e.g. 100 sigma. The clip
    // bounds the impact of such samples on the network input.
    //
    // Before we have 2 samples, std() returns 1.0; pass through unchanged in
    // that case to avoid feeding the network garbage scaled by a default std.
    std::vector<double> normalize(const std::vector<double>& obs, double clip = 10.0) const {
        if (obs.size() != dim_) {
            throw std::runtime_error("RunningMeanStd::normalize: obs size mismatch");
        }
        if (count_ < 2) {
            return obs;
        }
        const auto stds = std();
        std::vector<double> out(dim_);
        for (std::size_t i = 0; i < dim_; ++i) {
            double v = (obs[i] - mean_[i]) / stds[i];
            v = std::min(std::max(v, -clip), clip);
            out[i] = v;
        }
        return out;
    }

    // ---------------------------------------------------------------------
    // Save / load — a tiny plain-text format so policies + their normalizers
    // are inspectable and easy to diff. Format:
    //   dim=<n>
    //   count=<u64>
    //   mean=[v1, v2, ...]
    //   m2=[v1, v2, ...]
    // ---------------------------------------------------------------------
    void save(const std::string& path) const {
        auto out = open_artifact_output(path);
        out.setf(std::ios::scientific);
        out.precision(17);  // enough to round-trip double exactly.
        out << "dim=" << dim_ << '\n';
        out << "count=" << count_ << '\n';
        out << "mean=[";
        for (std::size_t i = 0; i < dim_; ++i) {
            if (i) out << ", ";
            out << mean_[i];
        }
        out << "]\n";
        out << "m2=[";
        for (std::size_t i = 0; i < dim_; ++i) {
            if (i) out << ", ";
            out << m2_[i];
        }
        out << "]\n";
    }

    static std::optional<RunningMeanStd> load(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) {
            // Missing file is non-fatal: the rollout falls back to passthrough.
            return std::nullopt;
        }
        // Local helpers — trim whitespace and parse "[v1, v2, ...]" forms.
        auto trim = [](std::string s) {
            const auto not_space = [](unsigned char c) { return !std::isspace(c); };
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
            s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
            return s;
        };
        auto parse_vector = [&](const std::string& body) {
            std::vector<double> out;
            const auto trimmed = trim(body);
            if (trimmed.size() < 2 || trimmed.front() != '[' || trimmed.back() != ']') {
                throw std::runtime_error("RunningMeanStd::load: expected vector value");
            }
            std::stringstream stream(trimmed.substr(1, trimmed.size() - 2));
            std::string item;
            while (std::getline(stream, item, ',')) {
                const auto t = trim(item);
                if (!t.empty()) out.push_back(std::stod(t));
            }
            return out;
        };

        // Read all key=value lines first; ordering in the file doesn't matter.
        std::size_t dim = 0;
        std::uint64_t count = 0;
        std::vector<double> mean;
        std::vector<double> m2;
        std::string line;
        while (std::getline(in, line)) {
            const auto trimmed = trim(line);
            if (trimmed.empty() || trimmed.front() == '#') continue;
            const auto eq = trimmed.find('=');
            if (eq == std::string::npos) continue;
            const auto key = trim(trimmed.substr(0, eq));
            const auto value = trimmed.substr(eq + 1);
            if (key == "dim") dim = static_cast<std::size_t>(std::stoul(value));
            else if (key == "count") count = static_cast<std::uint64_t>(std::stoull(value));
            else if (key == "mean") mean = parse_vector(value);
            else if (key == "m2") m2 = parse_vector(value);
        }

        if (dim == 0 || mean.size() != dim || m2.size() != dim) {
            throw std::runtime_error("RunningMeanStd::load: malformed file at " + path);
        }

        RunningMeanStd stats(dim);
        stats.mean_ = std::move(mean);
        stats.m2_ = std::move(m2);
        stats.count_ = count;
        return stats;
    }

   private:
    std::size_t dim_;
    std::vector<double> mean_;  // running per-dim mean
    std::vector<double> m2_;    // running sum of squared deviations (Welford M2)
    std::uint64_t count_;       // number of observations folded in
};

}  // namespace mujoco_rl_training
