"""兼容测试入口，实际测试用例位于测试目录。"""

import unittest
from pathlib import Path


def main() -> None:
    """发现并运行文案测试，将测试结果转换为进程退出码。"""
    test_directory = Path(__file__).resolve().parents[1] / "tests" / "unit"
    if not test_directory.is_dir():
        raise SystemExit(f"测试目录不存在：{test_directory}")
    suite = unittest.defaultTestLoader.discover(
        str(test_directory), pattern="test_copywriting_logic.py"
    )
    if suite.countTestCases() == 0:
        raise SystemExit("未发现文案测试，不能以零项测试认定通过")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
