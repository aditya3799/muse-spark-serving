"""Research-only route map. No automatic installation or weight-copy support.

Publication requires pre-reserved, byte-verified resident slots on all ranks.
The caller must quiesce execution and publish one coherent metadata generation.
Python guards run at capture, not replay; graph lifetime/invalidation is external.
"""
def map_core(ids, owners, counts, slots, rank, row_numbers, where):
    # Numpy and torch support these exact indexing operations.
    e=owners.shape[0]
    safe=ids.clip(0,e-1)
    chosen=owners[safe, row_numbers % counts[safe]]
    return where((ids>=0)&(ids<e)&(chosen==rank),slots[safe],-1)

class PreparedReplicaRoutes:
    def __init__(self, *, layer, owners, counts, slots, row_numbers, rank, torch_module):
        # Intentional fail-closed public constructor: storage/copy integration
        # and rank-coherent publication have not been implemented/qualified.
        raise RuntimeError('HELD: replica weight publication and graph lifetime gate required')

    def __call__(self, layer, ids):
        assert layer is self.layer
        assert ids.ndim==2 and ids.shape[0]<=self.row_numbers.shape[0]
        assert ids.device==self.owners.device
        return map_core(ids,self.owners,self.counts,self.slots,self.rank,
                        self.row_numbers[:ids.shape[0]],self.torch.where).contiguous()
