def _broadcast_reqs_across_ranks(
        self, recv_reqs: Optional[List], local_reqs: Optional[List] = None
    ) -> List:
        """local_reqs ride the work channel, which is scoped to the ranks
        sharing one waiting queue; the control channel fans out from global
        rank 0 and would overwrite every DP group's aborts but the first.
        """
        local_reqs = local_reqs or []
        if get_parallel().attn_dp_enabled:
            if get_parallel().attn_tp_rank == 0 and get_parallel().attn_cp_rank == 0:
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
                work_reqs.extend(local_reqs)
            else:
                work_reqs = None
                control_reqs = None

            work_reqs = attn_cp_tp_broadcast_pyobj(work_reqs)

            # When dp_attention_local_control_broadcast is enabled, each DP
            # group leader already receives control messages from the DP
            # controller, so we broadcast within attn_tp_group + attn_cp_group
            # instead of the full tp_group.  This avoids an expensive
            # all-ranks gloo sync.
            _local_ctrl = (
                get_parallel().enable_dp_attention_local_control_broadcast
                or get_exec().moe.is_ep_scale_joiner
            )
            if _local_ctrl:
                control_reqs = attn_cp_tp_broadcast_pyobj(control_reqs)
            elif get_parallel().tp_size != 1:
                tp_group = get_parallel().tp_group
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    tp_group.rank,
                    tp_group.cpu_group,
                    src=tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        else:
            if recv_reqs is not None:
                recv_reqs = [*recv_reqs, *local_reqs]
            if get_parallel().tp_size != 1:
                tp_group = get_parallel().tp_group
                recv_reqs = broadcast_pyobj(
                    recv_reqs,
                    tp_group.rank,
                    tp_group.cpu_group,
                    src=tp_group.ranks[0],
                )
        return recv_reqs