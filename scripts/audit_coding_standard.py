"""Check the PDF's mechanical rules without importing or running project modules.

Run ``python scripts/audit_coding_standard.py --check`` to reject violations.
The JSON report retains the PDF's 500-line recommendation as an advisory.
Black, isort, flake8, mypy, pylint and pytest supply the other quality gates.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import io
import json
import logging
import re
import tokenize
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OWNED_ROOTS = ("src", "scripts", "tests", "integration_tests")
EXTERNAL_METHODS = {"do_GET", "do_POST", "visit_AnnAssign"}
NAME_PATTERN = re.compile(r"_*[a-z][a-z0-9_]*|__[a-z][a-z0-9_]*__")
GOOGLE_RESULT = re.compile(r"^(Returns|Yields):", re.MULTILINE)


@dataclass(frozen=True)
class Finding:
    """One line-specific result, separate from nonblocking recommendations.

    Attributes:
        file: Project-relative owned source path.
        line: One-based source line.
        rule: Section of the supplied coding-standard PDF.
        detail: Actionable description of the finding.
        severity: Error for enforced rules or advisory for recommendations.
    """

    file: str
    line: int
    rule: str
    detail: str
    severity: str = "error"


def _finding(path: Path, node: ast.AST, rule: str, detail: str) -> Finding:
    return Finding(path.as_posix(), getattr(node, "lineno", 1), rule, detail)


def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    parameters = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    parameters += [arg for arg in (node.args.vararg, node.args.kwarg) if arg]
    return [arg for arg in parameters if arg.arg not in {"self", "cls"}]


def _interface_findings(
    path: Path, node: ast.FunctionDef | ast.AsyncFunctionDef
) -> list[Finding]:
    results = []
    parameters = _parameters(node)
    if len(parameters) > 5:
        results.append(_finding(path, node, "8.1", f"Too many parameters: {node.name}"))
    if not NAME_PATTERN.fullmatch(node.name) and node.name not in EXTERNAL_METHODS:
        results.append(_finding(path, node, "3.2", f"Non-snake-case name: {node.name}"))
    if node.name.startswith("_") and node.name != "__init__":
        return results
    docstring = ast.get_docstring(node) or ""
    missing_types = [arg.arg for arg in parameters if arg.annotation is None]
    if missing_types or node.returns is None:
        results.append(
            _finding(path, node, "6.1", f"Incomplete public annotations: {node.name}")
        )
    missing_docs = [
        arg.arg
        for arg in parameters
        if not re.search(
            rf"^\s*\*{{0,2}}{re.escape(arg.arg)}(?:\s*\([^\n]+\))?:",
            docstring,
            re.MULTILINE,
        )
    ]
    if missing_docs:
        results.append(
            _finding(path, node, "5.3", f"Missing Args for {node.name}: {missing_docs}")
        )
    if (
        node.returns is not None
        and ast.unparse(node.returns) not in {"None", "NoReturn"}
        and not GOOGLE_RESULT.search(docstring)
    ):
        results.append(_finding(path, node, "5.3", f"Missing Returns: {node.name}"))
    return results


def _handles_exception(node: ast.ExceptHandler) -> bool:
    if any(isinstance(child, ast.Raise) for child in ast.walk(node)):
        return True
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr in {"debug", "info", "warning", "error", "exception", "log"}
        for child in ast.walk(node)
    )


def _syntax_findings(path: Path, tree: ast.Module) -> list[Finding]:
    results = []
    bootstrap_line = next(
        (
            node.lineno
            for node in tree.body
            if isinstance(node, ast.Import)
            and any(alias.name == "run" for alias in node.names)
        ),
        None,
    )
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            results.extend(_interface_findings(path, node))
            defaults = node.args.defaults + node.args.kw_defaults
            if any(
                isinstance(value, (ast.List, ast.Dict, ast.Set)) for value in defaults
            ):
                results.append(_finding(path, node, "8.1", "Mutable default argument"))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                not node.name.startswith("_") or node.name == "__init__"
            ) and not ast.get_docstring(node):
                results.append(
                    _finding(path, node, "5.3", f"Missing docstring: {node.name}")
                )
        if isinstance(node, ast.Import) and len(node.names) > 1:
            results.append(
                _finding(path, node, "4.2", "Multiple modules in one import")
            )
        if isinstance(node, ast.ImportFrom):
            if (
                bootstrap_line is not None
                and node.lineno < bootstrap_line
                and (node.module or "").startswith("multimodal_ecommerce")
            ):
                results.append(
                    _finding(
                        path, node, "4.1", "Project import precedes path bootstrap"
                    )
                )
            if node.level or any(alias.name == "*" for alias in node.names):
                results.append(
                    _finding(path, node, "4.2", "Relative or wildcard import")
                )
        if isinstance(node, ast.ExceptHandler):
            if node.type is None or not _handles_exception(node):
                results.append(
                    _finding(path, node, "7.1", "Bare or unlogged exception handler")
                )
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Raise)
                    and isinstance(child.exc, ast.Call)
                    and child.cause is None
                ):
                    results.append(
                        _finding(
                            path, child, "7.3", "New exception needs an explicit cause"
                        )
                    )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if (
                path.parts[0] in {"tests", "integration_tests"}
                and node.func.id == "print"
            ):
                results.append(_finding(path, node, "10.2", "print in test"))
            if node.func.id in {"eval", "exec"} and not (
                path.as_posix() == "src/multimodal_ecommerce/pair_reranker.py"
                and node.func.id == "exec"
            ):
                results.append(
                    _finding(path, node, "9.2", "Review dynamic code execution")
                )
    return results


def _name_findings(path: Path, tree: ast.Module) -> list[Finding]:
    results = []
    if not NAME_PATTERN.fullmatch(path.stem):
        results.append(_finding(path, tree, "3.2", "Non-snake-case module name"))
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        name = (
            node.arg
            if isinstance(node, ast.arg)
            else (
                node.id
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                else None
            )
        )
        if name is None or name.startswith("_") or name not in vars(builtins):
            continue
        assignment = parents.get(node)
        container = parents.get(assignment) if assignment else None
        # TypedDict fields describe external JSON keys such as "type" and "format".
        if isinstance(assignment, ast.AnnAssign) and isinstance(
            container, ast.ClassDef
        ):
            if any(
                isinstance(base, ast.Name) and base.id == "TypedDict"
                for base in container.bases
            ):
                continue
        results.append(_finding(path, node, "3.1", f"Shadows builtin: {name}"))
    return results


def _token_findings(path: Path, source: str) -> list[Finding]:
    results = []
    previous = None
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if (
            token.type == tokenize.STRING
            and previous
            and previous.type == tokenize.STRING
        ):
            results.append(
                Finding(
                    path.as_posix(),
                    token.start[0],
                    "2.4",
                    "Implicit string concatenation",
                )
            )
        if token.type not in {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING}:
            previous = token
    return results


def scan_file(path: Path, root: Path = PROJECT_ROOT) -> list[Finding]:
    """Parse one owned source file without executing its imports.

    Args:
        path: Project-relative Python file to inspect.
        root: Project root containing that file.

    Returns:
        Mandatory violations and separately marked recommendations.
    """
    source = (root / path).read_text(encoding="utf-8-sig")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        logging.getLogger(__name__).debug("Cannot parse %s", path, exc_info=True)
        return [Finding(path.as_posix(), error.lineno or 1, "2", str(error))]
    results = _syntax_findings(path, tree) + _name_findings(path, tree)
    results.extend(_token_findings(path, source))
    if not ast.get_docstring(tree):
        results.append(_finding(path, tree, "5.3", "Missing module docstring"))
    if len(source.splitlines()) > 500:
        results.append(
            Finding(
                path.as_posix(),
                1,
                "8.3",
                "Module exceeds the suggested 500 lines",
                "advisory",
            )
        )
    return results


def main() -> int:
    """Audit all owned Python and write a reproducible, machine-readable report.

    Returns:
        Nonzero when --check is set and any mandatory violation remains.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/coding_standard_audit/findings.json"),
    )
    options = parser.parse_args()
    output = (
        options.output
        if options.output.is_absolute()
        else PROJECT_ROOT / options.output
    ).resolve()
    report_root = (PROJECT_ROOT / "artifacts/coding_standard_audit").resolve()
    if (
        not report_root.is_relative_to(PROJECT_ROOT.resolve())
        or not output.is_relative_to(report_root)
        or output.suffix != ".json"
    ):
        parser.error(
            "--output must be a JSON file inside artifacts/coding_standard_audit"
        )
    paths = sorted(
        path.relative_to(PROJECT_ROOT)
        for folder in OWNED_ROOTS
        for path in (PROJECT_ROOT / folder).rglob("*.py")
    )
    findings = [finding for path in paths for finding in scan_file(path, PROJECT_ROOT)]
    counts = Counter(finding.severity for finding in findings)
    report = dict(
        scope=list(OWNED_ROOTS),
        files=len(paths),
        mandatory_errors=counts["error"],
        advisories=counts["advisory"],
        status="enforced_checks_passed" if not counts["error"] else "violations_found",
        findings=[asdict(finding) for finding in findings],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Scanned {len(paths)} files: {counts['error']} errors, {counts['advisory']} advisories"
    )
    for finding in findings:
        if finding.severity == "error":
            print(f"{finding.file}:{finding.line}: {finding.rule} {finding.detail}")
    return int(options.check and counts["error"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
