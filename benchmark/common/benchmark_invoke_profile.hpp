// Per-op invoke profiler shared by netkit / TFLM MNIST profile benchmarks.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace BenchmarkInvokeProfile {

constexpr int kMaxProfileTags = 32;

class TagTickAggregator {
 public:
  static uint64_t NowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }

  void RecordLayer(const char* tag, uint64_t duration_ns) {
    AddTagNanoseconds(tag, duration_ns);
  }

  void AddWallMicros(double wall_us) { total_wall_us_ += wall_us; }

  void FinishInvoke() { ++num_invokes_; }

  int num_invokes() const { return num_invokes_; }

  double mean_wall_us() const {
    return num_invokes_ > 0 ? total_wall_us_ / static_cast<double>(num_invokes_) : 0.0;
  }

  double NanosToMicros(uint64_t ns) const {
    return static_cast<double>(ns) / 1000.0;
  }

  uint64_t TotalOpNanoseconds() const {
    uint64_t sum = 0;
    for (int i = 0; i < num_tags_; ++i) {
      sum += tag_ns_[i];
    }
    return sum;
  }

  double MeanOpMicrosPerInvoke() const {
    return num_invokes_ > 0
               ? NanosToMicros(TotalOpNanoseconds() /
                                 static_cast<uint64_t>(num_invokes_))
               : 0.0;
  }

  void PrintProfileSummaries(const char* runtime, const char* model) const {
    if (num_invokes_ <= 0) {
      return;
    }

    const double mean_wall = mean_wall_us();
    const double mean_op_us = MeanOpMicrosPerInvoke();
    const double gap_us = mean_wall - mean_op_us;
    const double gap_pct = mean_wall > 0.0 ? (gap_us / mean_wall) * 100.0 : 0.0;

    std::printf(
        "PROFILE_SUMMARY runtime=%s model=%s kind=meta tag=wall_clock "
        "mean_us=%.3f invoke_count=%d\n",
        runtime, model, mean_wall, num_invokes_);
    std::printf(
        "PROFILE_SUMMARY runtime=%s model=%s kind=meta tag=op_time_sum "
        "mean_us=%.3f invoke_count=%d\n",
        runtime, model, mean_op_us, num_invokes_);
    std::printf(
        "PROFILE_SUMMARY runtime=%s model=%s kind=meta tag=between_op_overhead "
        "mean_us=%.3f pct=%.1f invoke_count=%d\n",
        runtime, model, gap_us, gap_pct, num_invokes_);

    struct TagRow {
      const char* tag;
      uint64_t ns;
      double mean_us;
      double pct;
    };
    std::array<TagRow, kMaxProfileTags> rows{};
    int row_count = 0;
    for (int i = 0; i < num_tags_; ++i) {
      const uint64_t tag_mean_ns =
          tag_ns_[i] / static_cast<uint64_t>(num_invokes_);
      const double tag_mean_us = NanosToMicros(tag_mean_ns);
      const double pct = mean_wall > 0.0 ? (tag_mean_us / mean_wall) * 100.0 : 0.0;
      rows[row_count++] = TagRow{tag_names_[i], tag_ns_[i], tag_mean_us, pct};
    }

    std::sort(rows.begin(), rows.begin() + row_count,
              [](const TagRow& a, const TagRow& b) { return a.mean_us > b.mean_us; });

    for (int i = 0; i < row_count; ++i) {
      std::printf(
          "PROFILE_SUMMARY runtime=%s model=%s kind=op tag=%s mean_us=%.3f pct=%.1f "
          "invoke_count=%d\n",
          runtime, model, rows[i].tag, rows[i].mean_us, rows[i].pct, num_invokes_);
    }
  }

 private:
  void AddTagNanoseconds(const char* tag, uint64_t delta) {
    for (int i = 0; i < num_tags_; ++i) {
      if (std::strcmp(tag_names_[i], tag) == 0) {
        tag_ns_[i] += delta;
        return;
      }
    }
    if (num_tags_ >= kMaxProfileTags) {
      return;
    }
    tag_names_[num_tags_] = tag;
    tag_ns_[num_tags_] = delta;
    ++num_tags_;
  }

  const char* tag_names_[kMaxProfileTags]{};
  uint64_t tag_ns_[kMaxProfileTags]{};
  int num_tags_ = 0;

  int num_invokes_ = 0;
  double total_wall_us_ = 0.0;
};

}  // namespace BenchmarkInvokeProfile
