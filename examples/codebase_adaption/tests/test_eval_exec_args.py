"""Regression tests for the train->eval exec-args leak fix.

Context: swe_bench_cl (train) writes CLBENCH_SINGULARITY_EXEC_ARGS process-globally;
default_singularity_exec_args() reads it first, so codebase (eval) containers would
inherit SWECL's args. codebase_rollout._make_task's eval branch restores the pre-train
snapshot (_ORIG_SINGULARITY_EXEC_ARGS) so eval uses codebase's own defaults, while a
user-supplied explicit override is preserved (not clobbered). See
notes/EVAL_EXEC_ARGS_LEAK_FIX.md.

Must run inside the Miles dev SIF (needs clbench + miles + jinja2 on PYTHONPATH):

  apptainer exec --bind /data,/home/qixinx \
    /data/user_data/qixinx/images/miles_dev-202606081341.sif bash -c '
      export PYTHONPATH=/data/user_data/qixinx/miles_pydeps/codebase_py312_clean:\
/home/qixinx/continual-learning-bench:/home/qixinx/miles
      cd /home/qixinx/miles && python3 examples/codebase_adaption/tests/test_eval_exec_args.py'
"""
import os
import types

ENV = "CLBENCH_SINGULARITY_EXEC_ARGS"


def _testbed_on_path():
    from src.tasks.container_backend import default_singularity_exec_args
    return any("testbed" in x for x in default_singularity_exec_args())


def _has_fakeroot():
    from src.tasks.container_backend import default_singularity_exec_args
    return "--fakeroot" in default_singularity_exec_args()


def test_swecl_args_contract():
    """SWECL exec args must carry testbed PATH and must NOT carry --fakeroot."""
    from src.tasks.swe_bench_cl.task import _SWECL_SINGULARITY_EXEC_ARGS as SW
    assert "testbed" in SW and "--fakeroot" not in SW
    os.environ.pop(ENV, None)
    assert not _testbed_on_path() and _has_fakeroot(), "backend default = codebase (fakeroot, no testbed)"
    os.environ[ENV] = SW
    assert _testbed_on_path() and not _has_fakeroot(), "env-precedence: SWECL args win"
    os.environ.pop(ENV, None)


def _make_eval_task():
    """Call the REAL _make_task eval branch; returns True if it actually ran."""
    from examples.codebase_adaption import codebase_rollout as cr
    root = os.environ.get("CLBENCH_ROOT", "/home/qixinx/continual-learning-bench")
    ds = os.path.join(root, "data", "codebase_adaptation", "final-dataset.jsonl")
    if not os.path.exists(ds):
        return False
    args = types.SimpleNamespace(codebase_clbench_root=root, codebase_seed=42,
                                 codebase_max_steps_per_issue=3)
    try:
        cr._make_task(args, "heldout", ["dummy-1"], ["dummy"])  # eval branch
        return True
    except Exception as e:  # heldout id may not be constructable; env is set before that
        # _make_task sets the env at the top of the else branch, before any failure
        # deeper in task construction, so the env assertion below is still meaningful.
        print(f"  (note: _make_task raised after env reset: {type(e).__name__})")
        return True


def test_eval_restore_no_override():
    """No explicit override: train=SWECL, eval=codebase default (no leak), retrain=SWECL."""
    from src.tasks.swe_bench_cl.task import _SWECL_SINGULARITY_EXEC_ARGS as SW
    from examples.codebase_adaption import codebase_rollout as cr
    os.environ.pop(ENV, None)
    cr._ORIG_SINGULARITY_EXEC_ARGS = None  # snapshot taken at import when env was unset
    os.environ.setdefault(ENV, SW)                       # train
    assert _testbed_on_path()
    _make_eval_task()                                    # eval (real _make_task)
    assert not _testbed_on_path() and _has_fakeroot(), "eval must fall back to codebase default"
    os.environ.setdefault(ENV, SW)                       # next train
    assert _testbed_on_path(), "next train restores SWECL"
    os.environ.pop(ENV, None)


def test_eval_restore_preserves_override():
    """Explicit user override must survive train AND eval (not clobbered by pop)."""
    from src.tasks.swe_bench_cl.task import _SWECL_SINGULARITY_EXEC_ARGS as SW
    from examples.codebase_adaption import codebase_rollout as cr
    custom = "--contain --cleanenv --env FOO=bar"
    os.environ[ENV] = custom
    cr._ORIG_SINGULARITY_EXEC_ARGS = custom              # snapshot = user override
    os.environ.setdefault(ENV, SW)                       # train: setdefault keeps custom
    assert os.environ[ENV] == custom
    _make_eval_task()                                    # eval: must restore custom
    assert os.environ.get(ENV) == custom, "explicit override must be preserved"
    os.environ.pop(ENV, None)


if __name__ == "__main__":
    test_swecl_args_contract(); print("contract: PASS")
    test_eval_restore_no_override(); print("no-override state machine: PASS")
    test_eval_restore_preserves_override(); print("explicit-override state machine: PASS")
    print("ALL PASS")
