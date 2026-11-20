"""Preserve full router GEMM geometry while only publishing the owning rank's results."""
def apply(s,replace):
    old="        replicated_experts = any(\n"
    s=replace(s,old,"        self._replicated_humming_experts = replicated_experts = any(\n")
    s=replace(s,"        if sp_active:\n            hidden_states = layernorm_sp.sp_entry_scatter(hidden_states)",
              """        router_scratch = None
        if sp_active:
            if self._replicated_humming_experts:
                # One initialized full-layout router input per forward. Other ranks'
                # rows are independent and their local logits are never consumed.
                router_scratch = hidden_states.clone(memory_format=torch.contiguous_format)
            hidden_states = layernorm_sp.sp_entry_scatter(hidden_states)""")
    s=replace(s,"            stream, attn_input = layer(positions, stream, attn_input, forward_batch)",
                """            stream, attn_input = layer(
                positions, stream, attn_input, forward_batch, router_scratch=router_scratch
            )""")
    s=replace(s,"        forward_batch: ForwardBatch,\n    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:",
                "        forward_batch: ForwardBatch,\n        router_scratch: Optional[torch.Tensor] = None,\n    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:")
    s=replace(s,"            ffn_output = self.mlp(ffn_input, router_input, forward_batch)",
                "            ffn_output = self.mlp(ffn_input, router_input, forward_batch, router_scratch=router_scratch)")
    for nextline in ("        if (\n            forward_batch is not None","        ar_workspace = self.finalize_ar_workspace"):
        a="        forward_batch: Optional[ForwardBatch] = None,\n    ) -> torch.Tensor:\n"+nextline
        b="        forward_batch: Optional[ForwardBatch] = None,\n        router_scratch: Optional[torch.Tensor] = None,\n    ) -> torch.Tensor:\n"+nextline
        s=replace(s,a,b)
    s=replace(s,"        return self._forward_tp(hidden_states, router_input, forward_batch)",
                "        return self._forward_tp(hidden_states, router_input, forward_batch, router_scratch=router_scratch)")
    s=replace(s,"            router_input,\n            ar_workspace=ar_workspace,",
                "            router_input,\n            ar_workspace=ar_workspace,\n            router_scratch=router_scratch,")
    s=replace(s,"        *,\n        ar_workspace=None,\n    ) -> torch.Tensor:",
                "        *,\n        ar_workspace=None,\n        router_scratch: Optional[torch.Tensor] = None,\n    ) -> torch.Tensor:")
    a="""            router_logits = (
                self.gate.forward_split2(hidden_states)
                if use_split2
                else self.gate(router_input)
            )"""
    b="""            if router_scratch is not None:
                rows = hidden_states.shape[0]
                start = get_parallel().tp_group.rank_in_group * rows
                router_scratch.narrow(0, start, rows).copy_(hidden_states)
                router_logits = self.gate.forward_split2(router_scratch).narrow(0, start, rows)
            else:
                router_logits = (
                    self.gate.forward_split2(hidden_states)
                    if use_split2
                    else self.gate(router_input)
                )"""
    return replace(s,a,b)

