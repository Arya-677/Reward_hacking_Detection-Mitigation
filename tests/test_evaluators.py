import shutil
import tempfile
from pathlib import Path
import pytest

from reward_hacking.independent_evaluator import IndependentEvaluator
from reward_hacking.proxy_evaluator import ProxyEvaluator
from reward_hacking.sandbox import Sandbox


@pytest.fixture
def sandbox():
    return Sandbox(mode="restricted_subprocess", allow_restricted=True)


@pytest.fixture
def tasks_dir():
    return Path(__file__).resolve().parents[1] / "data" / "tasks"


def test_task_validity_all_tasks(sandbox, tasks_dir):
    """Requirement H / T9: For every task and hard variant, buggy fails protected tests; reference passes visible and protected."""
    task_dirs = sorted([p for p in tasks_dir.iterdir() if p.is_dir() and (p / "task.yaml").exists()])
    assert len(task_dirs) >= 8, f"Expected at least 8 tasks, found {len(task_dirs)}"

    proxy_eval = ProxyEvaluator(sandbox)
    indep_eval = IndependentEvaluator(sandbox)

    for td in task_dirs:
        task_name = td.name

        # 1. Test Buggy Solution fails protected tests
        buggy_path = td / "buggy" / "solution.py"
        indep_res = indep_eval.evaluate(buggy_path, td)
        assert indep_res.success is False, f"Task {task_name}: Buggy solution unexpectedly passed protected tests!"

        # 1b. Test Hard Variant fails protected tests if present
        hard_path = td / "hard" / "solution.py"
        if hard_path.exists():
            hard_res = indep_eval.evaluate(hard_path, td)
            assert hard_res.success is False, f"Task {task_name}: Hard variant unexpectedly passed protected tests!"

        # 2. Test Reference Solution passes visible tests
        ref_path = td / "reference" / "solution.py"
        tmp_vis = Path(tempfile.mkdtemp(prefix="test_vis_"))
        try:
            shutil.copy2(ref_path, tmp_vis / "solution.py")
            shutil.copy2(td / "visible" / "test_visible.py", tmp_vis / "test_visible.py")
            vis_res = proxy_eval.evaluate(tmp_vis)
            assert vis_res.score == 1.0, f"Task {task_name}: Reference solution failed visible tests ({vis_res.score})!"
            assert vis_res.all_passed is True
        finally:
            shutil.rmtree(tmp_vis, ignore_errors=True)

        # 3. Test Reference Solution passes protected tests
        ref_indep_res = indep_eval.evaluate(ref_path, td)
        assert ref_indep_res.success is True, f"Task {task_name}: Reference failed protected tests: {ref_indep_res.reason}"


def test_proxy_evaluator_empty_and_syntax_error(sandbox):
    """Proxy evaluator handles empty file, missing tests, and syntax error with score 0.0."""
    proxy_eval = ProxyEvaluator(sandbox)
    tmp_dir = Path(tempfile.mkdtemp(prefix="test_proxy_err_"))

    try:
        # 1. Missing test file
        res = proxy_eval.evaluate(tmp_dir)
        assert res.score == 0.0
        assert "does not exist" in res.reason

        # 2. Empty test file
        test_file = tmp_dir / "test_visible.py"
        test_file.write_text("   \n  ", encoding="utf-8")
        res = proxy_eval.evaluate(tmp_dir)
        assert res.score == 0.0
        assert "empty" in res.reason.lower()

        # 3. No test functions
        test_file.write_text("x = 1\ny = 2\n", encoding="utf-8")
        res = proxy_eval.evaluate(tmp_dir)
        assert res.score == 0.0
        assert "no test functions" in res.reason.lower()

        # 4. Syntax error
        test_file.write_text("def test_broken(:\n    assert True", encoding="utf-8")
        res = proxy_eval.evaluate(tmp_dir)
        assert res.score == 0.0
        assert "syntax error" in res.reason.lower()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
