#include "novaphy/io/feature_completeness.h"

#include <stdexcept>

namespace novaphy {

FeatureCheckReport FeatureCompletenessChecker::run_check() const {
    FeatureCheckReport report;
    report.items = {
        {"continuous_collision_detection", false, "not_implemented"},
        {"joint_motor", false, "not_implemented"},
        {"soft_rigid_coupling", true, "libuipc"},
        {"gpu_parallel_solver", true, "libuipc_cuda"},
    };
    report.all_aligned = true;
    for (const FeatureCheckItem& item : report.items) {
        if (!item.available) {
            report.all_aligned = false;
            break;
        }
    }
    return report;
}

void FeatureCompletenessChecker::require_full_alignment() const {
    const FeatureCheckReport report = run_check();
    if (!report.all_aligned) {
        throw std::runtime_error("Feature completeness check failed against Newton baseline.");
    }
}

}  // namespace novaphy
