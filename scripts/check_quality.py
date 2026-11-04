"""Run the complete read-only source-quality and regression gate.

Use the active project's Python interpreter. Reports go to
artifacts/coding_standard_audit; source files are never autoformatted here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/coding_standard_audit"
SCOPE = ("src", "scripts", "tests", "integration_tests")
CHECKS = {
    "black": ["-m", "black", "--check", *SCOPE],
    "isort": ["-m", "isort", "--check-only", *SCOPE],
    "flake8": ["-m", "flake8", *SCOPE],
    "mypy": ["-m", "mypy", "--no-incremental", *SCOPE],
    "pylint": ["-m", "pylint", "--errors-only", *SCOPE],
    "pdf_rules": ["scripts/audit_coding_standard.py", "--check"],
    "pytest": ["-m", "pytest", "-q"],
}


def main() -> int:
    """Check every owned Python file and preserve the evidence of each tool.

    Returns:
        Zero only when all checks, including tests, finish successfully.
    """
    environment = os.environ.copy()
    environment["PATH"] = (
        str(ROOT / ".tools/bin") + os.pathsep + environment.get("PATH", "")
    )
    if shutil.which("node", path=environment["PATH"]) is None:
        print("Node is required so browser-export tests cannot silently skip.")
        return 1
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logs = OUTPUT / "checks"
    logs.mkdir(exist_ok=True)
    results = {}
    for name, arguments in CHECKS.items():
        print(f"Checking {name} ...", flush=True)
        result = subprocess.run(
            [sys.executable, *arguments],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        transcript = result.stdout + result.stderr
        log = logs / f"{name}.log"
        log.write_text(transcript, encoding="utf-8")
        # Missing optional infrastructure must not count as tested behavior.
        passed = result.returncode == 0 and not (
            name == "pytest" and " skipped" in transcript
        )
        results[name] = dict(
            passed=passed, exit_code=result.returncode, log=str(log.relative_to(ROOT))
        )
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
        if not passed:
            print(transcript[-12000:])
    files = sorted(path for folder in SCOPE for path in (ROOT / folder).rglob("*.py"))
    report_path = OUTPUT / "findings.json"
    audit = json.loads(report_path.read_text()) if report_path.exists() else {}
    passed = all(result["passed"] for result in results.values())
    report = dict(
        checked_at=datetime.now(timezone.utc).isoformat(),
        status="enforced_checks_passed" if passed else "checks_failed",
        python=sys.version,
        scope=list(SCOPE),
        file_count=len(files),
        mandatory_errors=audit.get("mandatory_errors"),
        advisories=audit.get("advisories"),
        checks=results,
        versions={
            name: importlib.metadata.version(name)
            for name in CHECKS
            if name != "pdf_rules"
        },
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        limitations=[
            "The 500-line limit is a PDF recommendation, reported as an advisory.",
            "Static checks and regression tests do not prove every semantic or security property.",
            "GPU/model integration requires the local model assets; this gate runs offline regression tests.",
            "Historical receipts retain their original source hashes; new experiment releases require a new source freeze.",
        ],
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
