#pragma once

#include <string>

#include "novaphy/novaphy_config.h"

namespace novaphy {

/**
 * @brief Returns the semantic version string of the NovaPhy library.
 * @return Version string in the form `major.minor.patch`.
 */
inline std::string version() { return NOVAPHY_VERSION; }

inline std::string libuipc_bind_type() {
#if ! NOVAPHY_WITH_IPC
    return "none";
#elif NOVAPHY_BUNDLE_UIPC
    return "bundled";
#else
    return "system";
#endif
}

}  // namespace novaphy
