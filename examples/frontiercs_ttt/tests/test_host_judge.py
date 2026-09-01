from pathlib import Path

from examples.frontiercs_ttt.host_judge import (
    _default_frontiercs_root,
    _patch_host_testlib,
    _service_root,
)


def test_host_testlib_patch_is_exact_and_keeps_repository_source_unchanged():
    frontiercs_root = _default_frontiercs_root()
    assert frontiercs_root is not None
    path = Path(frontiercs_root) / "algorithmic" / "judge" / "src" / "gojudge.js"
    source = path.read_text(encoding="utf-8")
    patched = _patch_host_testlib(source)
    assert patched != source
    assert patched.count("'testlib.h': { content: testlibSource }") == 2
    assert patched.count("'-I', '.', '-o'") == 2
    assert patched.count("args: [CXX") >= 2
    assert path.read_text(encoding="utf-8") == source


def test_service_root_is_namespaced_under_configured_state_base(tmp_path):
    root = _service_root(18081, tmp_path)
    assert root.parent == tmp_path.resolve()
    assert root.name.endswith("-18081")
