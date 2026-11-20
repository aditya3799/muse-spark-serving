from pathlib import Path
src=Path('/mnt/c/Windows/Temp/finalize_projection_checks.py').read_text()
src=src.replace("research_iteration_2026-11-15/projection_candidate'", "research_iteration_2026-11-15/projection_fp32_candidate'")
src=src.replace("('build_projection_candidate.py','validate_projection_candidate.py','finalize_projection_checks.py')", "('build_projection_candidate.py','build_projection_fp32_candidate.py','validate_projection_fp32.py','benchmark_projection_fp32.py','check_projection_fp32_patch.py')")
exec(compile(src,__file__,'exec'))
