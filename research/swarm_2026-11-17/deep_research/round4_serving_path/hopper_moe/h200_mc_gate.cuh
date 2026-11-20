// Future H200-only test entry points. Never loaded by active serving code.
#pragma once
#include "mc_ready_plane_handoff.cuh"
#include "norm_residual.cuh"
namespace sglang {
inline int64_t research_local_pull_address(CommunicatorRef comm) {
 const auto& pull=comm->get_pull_obj();
 return reinterpret_cast<int64_t>(pull.get_workspace<8>(0).workspaces[pull.rank]);
}
}
