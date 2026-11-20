# DFlash FC row projection research integration

This is an unapplied diagnostic candidate over frozen release SHA256
`8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b`.
It does not establish a serving speedup or a >2x result. No submission was made
from this research lane. The current candidate hash is in each build's
`manifest.json`; rebuilding intentionally changes it when reviewed source changes.

## What executes

The ordinary, bias-free BF16 DFlash context FC is replicated across draft TP
ranks. For a registered, TP-divisible large M, each rank computes its own rows
using all K=40960 values and all N=8192 columns, then gathers raw FC rows. The
original full-M hidden normalization, each rank's distinct KV head weights,
RoPE, physical cache locations and prefix-valid commit lengths execute unchanged.
Eight global KV heads are eight different weights; gathering locally projected
KV rows instead would mix head owners. That earlier proposal is invalid.

The local Lt algorithm is discovered from the full-M operation at runtime,
checked with the vendor's local-shape legality API, and calibrated against the
actual checkpoint and native operation. There are no embedded workload shapes,
algorithm IDs or weight contents. Multiple input styles and every local row
partition must match native BF16 bits. Reduced-BF16 policies requiring an
unimplemented Lt preference mask decline the candidate. This finite calibration
is not a mathematical proof for every future activation.

The router's empirical mode is the diagnostic fast path. Its separate strict
shadow mode also computes the native full operation and selects native results
if bits disagree. Shadow timings are never used to claim a fast-path gain.

## Startup and activation

`build_integration.py --enable-row` builds `integration_row/candidate.diff` with
the row feature default-on and output-N default-off. Both remain independently
disableable with `SGLANG_EXPERIMENTAL_DFLASH_ROW_FC` and
`SGLANG_EXPERIMENTAL_DFLASH_OUTPUT_N`. An optional joint diagnostic is built with
`--enable-row --enable-n --name integration_joint`. The standalone helper's
default is off; activation is explicit in the selected built candidate.

The worker initializes adapters once from `prewarm_sampling`, under the real
serving forward stream. Scheduler ordering is initial KV pool allocation,
initial target/draft graph capture, this prewarm, then final
`post_capture_resize_kv_pool`. Therefore qualification is **not** before all KV
allocation. New scratch is retained on the actual serving stream and transient
qualification buffers are released before final pool sizing. Existing captured
target transformer graphs do not incorporate the later context-FC append hook.

Prefill rows come from configured capture buckets plus the configured chunk
size, filtered by generic matrix/TP/memory eligibility. The append method trims
target padding before projection, so a raw tail such as753 rows is not made into
a registered768-row call by target graph padding. Such tails remain native.
The large complete prefill chunks are the intended eligible work. Decode rows
come from configured decode batches times draft block size; unsupported rows
remain native. Row and output-N partition the same FC operation and their
component gains cannot be multiplied.

Startup includes coherent CPU-group preflight of complete shape sequence,
compile availability, calibration acceptance, recoverable allocation failures,
and actual transport correctness. A discovery exception on one rank cannot
skip the acceptance collective. Before enabling each shape, the row helper
also measures native FC versus local FC plus the real group gather; every rank
must show a lower combined median. Gather allocation uses the coordinator's
symmetric-memory context where enabled. A poisoned CUDA context remains a
fatal startup error, not a fallback promise.

## Memory and graph lifetime

The explicit calibration estimate is capped at1.8 billion bytes and accounts
for one full input, one native reference, one local output, three workspaces
and32MiB margin. It never retains two full input examples. Transport checks
release the full input before allocating gathered output. Timed native and
gathered outputs do not coexist. One workspace is shared across shapes on a
stream; different streams do not alias it.

The single-GPU package probe measured incremental allocated1,686,504,960 bytes
and reserved1,713,373,184 bytes while qualifying M512/M16384, with8,519,680 bytes
of retained scratch. That measurement is local RTX3050 evidence, not a bound on
H200 NCCL or symmetric-memory internals. The full chain fixture's3.54GiB peak
includes its model/test allocations and was never a judge incremental bound.

The production helper preserves global allocator peak history. Its logs include
current/peak before and after, with peak-after-minus-current-before labelled an
upper bound that includes prior history. Only the standalone measurement process
resets its own peak counter.

Registered plans are guarded by stream, shape, tensor layout, math policy and
weight identity/version. A local guard rejection still gathers that rank's
native row slice so peers retain the collective sequence. Inference tensors do
not have a version counter: checkpoint reload or math-policy changes require
destroying dependent graphs and rebuilding adapters. `close()` rejects a router
used in capture until graph destruction is explicitly acknowledged, then
synchronizes before releasing native plans/workspaces.

## Evidence and remaining gate

- `real_row_pilot.json` and `real_row_holdout.json`: independent seeds, actual
  recovered checkpoint FC/KV/norm, every FC row shard reconstructed, unchanged
  native full-M normalization and five KV/RoPE/scatter stages, sparse physical
  locations, changed graph inputs. All five tested row sizes passed.
- Their timings include one rank's FC, a full output staging copy and original
  tail. Peer FC reconstruction is outside timing and NCCL is absent. These are
  local compute/copy envelopes only.
- `row_adapter_cpu_tp2.json`, `row_adapter_cpu_tp4.json`: real Gloo communication
  exercises asymmetric failures and fallback sequencing with mocked native plans.
- `row_package_probe.json`: actual packaged native plans and checkpoint weights,
  graph replay/lifetime, unknown-stream fallback and startup allocation. CPU
  votes and peer row outputs are mocked on the single GPU.
- `integration_contract.json`: source-extracted model/worker methods execute on
  CPU, preserving normalization, padding trim, positions, sparse slots and
  prefix-valid commit semantics. It does not import/start the full engine.

A true H200 TP8 startup must still qualify vendor plans, transport ownership,
memory and latency, and the full model-output check must pass. A serving profile
must then establish that this FC stage has enough wall-time share to matter.
Even a hypothetical8x FC component gain needs about43.6% of total elapsed time
to lift the1.2363x parent to2x under a simple Amdahl model. Nothing in the local
component timings establishes that share.
