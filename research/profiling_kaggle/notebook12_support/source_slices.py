"""AST-selected init and pre-alignment statements; no real aligner/GEMM execution."""
import torch
SMALL_NUMEL_LIMIT=64

def _map_topk_ids_to_local_experts(
    topk_ids: torch.Tensor, local_expert_mapping: torch.Tensor
) -> torch.Tensor:
    """Map global expert ids while preserving the ``-1`` drop sentinel."""
    return torch.where(
        topk_ids < 0,
        -1,
        local_expert_mapping[topk_ids.clamp_min(0)],
    )

class BaselineSourceSlice(torch.nn.Module):
    def __init__(self, num_experts=256, num_local_experts=32, local_expert_offset=64):
        super().__init__()
        self.intermediate_size_per_partition=32
        expert_map = None
        if num_local_experts != num_experts:
            start = local_expert_offset
            expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
            expert_map[start : start + num_local_experts] = torch.arange(
                num_local_experts, dtype=torch.int32
            )
        self.register_buffer("_expert_map", expert_map, persistent=False)

    def forward(self, hidden_states, topk_scores, topk_ids):
        hidden_states = hidden_states.contiguous()
        num_tokens, size_k = hidden_states.shape
        topk = topk_ids.shape[1]
        size_n = self.intermediate_size_per_partition
        gemm1_n = 2 * size_n
        expert_map = self._expert_map
        fuse_mapping = (
            expert_map is not None
            and hidden_states.is_cuda
            and topk_ids.is_contiguous()
            and expert_map.is_contiguous()
            and 0 < topk_ids.numel() <= SMALL_NUMEL_LIMIT
        )
        if fuse_mapping:
            local_topk_ids = torch.empty_like(topk_ids, dtype=torch.int32)
        else:
            local_topk_ids = topk_ids.to(torch.int32)
            if expert_map is not None:
                local_topk_ids = _map_topk_ids_to_local_experts(
                    local_topk_ids, expert_map
                )
            local_topk_ids = local_topk_ids.contiguous()
        topk_weights = topk_scores.to(torch.float32).contiguous()
        return fuse_mapping, local_topk_ids, topk_weights

class CandidateSourceSlice(torch.nn.Module):
    def __init__(self, num_experts=256, num_local_experts=32, local_expert_offset=64):
        super().__init__()
        self.intermediate_size_per_partition=32
        expert_map = None
        if num_local_experts != num_experts:
            start = local_expert_offset
            expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
            expert_map[start : start + num_local_experts] = torch.arange(
                num_local_experts, dtype=torch.int32
            )
        self.register_buffer("_expert_map", expert_map, persistent=False)
        expert_lookup = (
            None
            if expert_map is None
            else torch.cat((expert_map.new_full((1,), -1), expert_map))
        )
        self.register_buffer("_expert_lookup", expert_lookup, persistent=False)

    def forward(self, hidden_states, topk_scores, topk_ids):
        hidden_states = hidden_states.contiguous()
        num_tokens, size_k = hidden_states.shape
        topk = topk_ids.shape[1]
        size_n = self.intermediate_size_per_partition
        gemm1_n = 2 * size_n
        expert_map = self._expert_map
        fuse_mapping = (
            expert_map is not None
            and hidden_states.is_cuda
            and topk_ids.is_contiguous()
            and expert_map.is_contiguous()
            and 0 < topk_ids.numel() <= SMALL_NUMEL_LIMIT
        )
        if fuse_mapping:
            local_topk_ids = torch.empty_like(topk_ids, dtype=torch.int32)
        else:
            local_topk_ids = topk_ids.to(torch.int32)
            if expert_map is not None:
                local_topk_ids = self._expert_lookup[
                    local_topk_ids.clamp_min(-1) + 1
                ]
            local_topk_ids = local_topk_ids.contiguous()
        topk_weights = topk_scores.to(torch.float32).contiguous()
        return fuse_mapping, local_topk_ids, topk_weights

    @torch.no_grad()
    def refresh_expert_lookup_(self) -> None:
        """Refresh after map edits, before consumers on the ordered CUDA stream.

        Preserves captured storage. Replacing/moving buffers requires graph
        recapture; direct map mutations are not automatically detected.
        """
        if self._expert_map is not None:
            self._expert_lookup[0].fill_(-1)
            self._expert_lookup[1:].copy_(self._expert_map)

    @torch.no_grad()
    def update_expert_map_(self, mapping: torch.Tensor) -> None:
        """Update fixed-shape routing metadata without replacing graph inputs."""
        if self._expert_map is None:
            raise ValueError("Mapping cannot be enabled without reconstructing the module")
        if (
            mapping.shape != self._expert_map.shape
            or mapping.dtype != torch.int32
            or mapping.device != self._expert_map.device
        ):
            raise ValueError("Expert map update must preserve shape, int32 dtype and device")
        self._expert_map.copy_(mapping)
        self.refresh_expert_lookup_()

from typing import Optional

class SharedSourceSlice(torch.nn.Module):
    def __init__(self,num_experts=256,num_local_experts=32,local_expert_offset=64):
        super().__init__()
        self.intermediate_size_per_partition=32
        expert_map = None
        if num_local_experts != num_experts:
            start = local_expert_offset
            expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
            expert_map[start : start + num_local_experts] = torch.arange(
                num_local_experts, dtype=torch.int32
            )
        expert_lookup = (
            None
            if expert_map is None
            else torch.cat((expert_map.new_full((1,), -1), expert_map))
        )
        self.register_buffer("_expert_lookup", expert_lookup, persistent=False)

    @property
    def _expert_map(self) -> Optional[torch.Tensor]:
        """View of the single owning lookup; stays aliased after module moves."""
        if self._expert_lookup is None:
            return None
        return self._expert_lookup[1:]

    @torch.no_grad()
    def update_expert_map_(self, mapping: torch.Tensor) -> None:
        """One metadata copy; the mapping view and lookup share storage.

        Update before consumers on an ordered stream. Replacing/moving the
        owning buffer requires graph recapture. Map replacement is unsupported.
        """
        expert_map = self._expert_map
        if expert_map is None:
            raise ValueError("Mapping cannot be enabled without reconstructing the module")
        if (
            mapping.shape != expert_map.shape
            or mapping.dtype != torch.int32
            or mapping.device != expert_map.device
        ):
            raise ValueError("Expert map update must preserve shape, int32 dtype and device")
        expert_map.copy_(mapping)

    def forward(self,hidden_states,topk_scores,topk_ids):
        hidden_states = hidden_states.contiguous()
        num_tokens, size_k = hidden_states.shape
        topk = topk_ids.shape[1]
        size_n = self.intermediate_size_per_partition
        gemm1_n = 2 * size_n
        expert_map = self._expert_map
        fuse_mapping = (
            expert_map is not None
            and hidden_states.is_cuda
            and topk_ids.is_contiguous()
            and expert_map.is_contiguous()
            and 0 < topk_ids.numel() <= SMALL_NUMEL_LIMIT
        )
        if fuse_mapping:
            local_topk_ids = torch.empty_like(topk_ids, dtype=torch.int32)
        else:
            local_topk_ids = topk_ids.to(torch.int32)
            if expert_map is not None:
                local_topk_ids = self._expert_lookup[
                    local_topk_ids.clamp_min(-1) + 1
                ]
            local_topk_ids = local_topk_ids.contiguous()
        topk_weights = topk_scores.to(torch.float32).contiguous()
        return fuse_mapping,local_topk_ids,topk_weights
