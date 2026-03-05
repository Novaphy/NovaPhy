#pragma once

#include <string>
#include <vector>

namespace novaphy {

struct FeatureCheckItem {
    std::string name;
    bool available = false;
    std::string backend;
};

struct FeatureCheckReport {
    std::vector<FeatureCheckItem> items;
    bool all_aligned = false;
};

class FeatureCompletenessChecker {
public:
    FeatureCheckReport run_check() const;
    void require_full_alignment() const;
};

}  // namespace novaphy
