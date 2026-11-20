# Full-width projection tuner

This optional inference component wraps an existing unquantized, bias-free PyTorch Linear. Runtime cuBLASLt heuristics supply the reduction and eligible algorithm choices. The source contains no model dimensions, rank assumptions or frozen algorithm IDs. Compile the source once in the deployment environment with its actual CUDA library.

~~~bash
python full_projector/build.py
~~~

After checkpoint loading, create one wrapper for the existing layer, call prepare with distinct representative warmup and qualification tensors on its inference stream, and keep the existing hidden normalization after the wrapper. Unregistered shapes and unsupported paths execute the existing layer.

~~~python
from full_projector import CompleteProjection

projection = CompleteProjection(draft_model.fc.eval(), empirical=False)
with torch.inference_mode():
    qualification = projection.prepare(tuning_inputs, separate_validation_inputs)
    normalized = draft_model.hidden_norm(projection(target_hidden))
~~~

Default strict mode checks all output bits on every call and preserves the original result on divergence. Empirical mode skips the second GEMM, so it relies on the finite device-specific qualification. Inputs supplied to prepare must be representative of the loaded checkpoint and should include difficult cancellation cases. Qualification creates its own vendor handle and workspace; startup wall time, paired native/optimized times, and the number of calls to recover the setup cost are returned. Unsupported cuBLAS/PyTorch builds naturally remain on the native layer when its vendor workspace cannot be resolved. Call close after destroying dependent CUDA graphs. Invalidate and prepare again after any in-place weight reload, especially for inference tensors without version counters.

This source-only prototype was developed from local SM86 single-GPU tests. Native SM90 H200 comparison, full-model greedy output validation, deployment cold start and peak memory are outstanding. No hardware-specific configuration should be copied to a different device or CUDA installation.
