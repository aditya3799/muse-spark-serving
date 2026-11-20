from pathlib import Path
p=Path(__file__).with_name("fixed_tree_adapter.py")
s=p.read_text(encoding="utf-8-sig")
at="    def propose("
if "    def active(" not in s:
    s=s.replace(at,'''    def active(self, batch):
        """View a single allocator-owned slab across all eager batch sizes."""
        if not 0 < batch <= self.batch:
            raise ValueError("Batch exceeds the slab capacity")
        view = object.__new__(type(self))
        view.batch = batch
        for key,value in vars(self).items():
            if key == "batch":
                continue
            if key in ("cu_q","cu_k"):
                rows = batch * WIDTH + 1
            elif key in ("ancestors","ancestor_lens","lengths","positions","full","swa"):
                rows = batch * WIDTH
            else:
                rows = batch
            setattr(view,key,value[:rows])
        return view

'''+at)
s=s.replace("table.shape[1] != self.prefix_pages.shape[1]","table.shape[1] < self.prefix_pages.shape[1]")
s=s.replace("torch.index_select(table, 0, req_ids, out=self.prefix_pages)","torch.index_select(table[:, :self.prefix_pages.shape[1]], 0, req_ids, out=self.prefix_pages)")
p.write_text(s,encoding="utf8")

