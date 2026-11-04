"""Exercise the standard gate's failure exit and distinction between rule levels."""

from __future__ import annotations

from pathlib import Path

import pytest

import audit_coding_standard as audit


@pytest.mark.parametrize(
    "output",
    [
        "../escape.json",
        "src/overwrite.json",
        "artifacts/coding_standard_audit/report.py",
    ],
)
def test_report_path_cannot_escape_its_json_directory(
    output: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject traversal and source destinations before writing any report.

    Args:
        output: Deliberately invalid destination supplied on the command line.
        tmp_path: Isolated root that must remain empty after rejection.
        monkeypatch: Restore command arguments and project-root configuration.
    """
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["audit_coding_standard.py", "--output", output])
    with pytest.raises(SystemExit) as error:
        audit.main()
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_noncompliant_source_fails_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a missing interface contract rather than merely writing a report.

    Args:
        tmp_path: Isolated project root for the deliberately invalid source.
        monkeypatch: Restore command arguments and project-root configuration.
    """
    source = tmp_path / "src"
    source.mkdir()
    (source / "example.py").write_text("def undocumented(value):\n    return value\n")
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(audit, "OWNED_ROOTS", ("src",))
    monkeypatch.setattr("sys.argv", ["audit_coding_standard.py", "--check"])
    assert audit.main() == 1
    assert (tmp_path / "artifacts/coding_standard_audit/findings.json").is_file()


def test_long_module_is_only_a_pdf_recommendation(tmp_path: Path) -> None:
    """Keep the PDF's recommended module size separate from mandatory failures.

    Args:
        tmp_path: Isolated directory containing a documented long module.
    """
    path = Path("example.py")
    (tmp_path / path).write_text(
        '"""A documented module."""\n' + "# Explanation.\n" * 501
    )
    findings = audit.scan_file(path, tmp_path)
    assert len(findings) == 1
    assert findings[0].rule == "8.3" and findings[0].severity == "advisory"


def test_implicit_concatenation_and_mutable_defaults_are_rejected(
    tmp_path: Path,
) -> None:
    """Catch syntactically valid hazards that formatting tools can leave intact.

    Args:
        tmp_path: Isolated directory for the invalid private helper.
    """
    path = Path("example.py")
    (tmp_path / path).write_text(
        "\n".join(
            [
                '"""A documented module."""',
                "def _private(items=[]):",
                '    return ("a" "b")',
            ]
        )
        + "\n"
    )
    rules = {finding.rule for finding in audit.scan_file(path, tmp_path)}
    assert {"2.4", "8.1"} <= rules
