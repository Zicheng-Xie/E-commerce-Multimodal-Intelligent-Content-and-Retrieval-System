"""Public-only runtime manifests, content-bound caches and scoped acceptance access."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path, sha256_file


def digest(value: object) -> str:
    """Fingerprint canonical JSON without depending on filesystem locations.

    Args:
        value: JSON-compatible data.

    Returns:
        Canonical SHA256 identity.
    """
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def load(path: Path) -> Any:
    """Load JSON at a boundary whose access policy has already been checked.

    Args:
        path: JSON file.

    Returns:
        Parsed JSON document.
    """
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: object) -> None:
    """Replace an owned output only after its JSON has been fully serialized.

    Args:
        path: Output inside the project.
        value: JSON-compatible output.
    """
    path = ensure_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class RuntimeAccess:
    """Keep ordinary development away from reserved data and private evaluator files."""

    def __init__(
        self, ticket: Path | None = None, credential: str | None = None
    ) -> None:
        """Bind an optional single-run authorization, never a general inference override.

        Args:
            ticket: Existing acceptance claim created after all three freezes.
            credential: Matching credential supplied only to this run's model worker.
        """
        self.registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
        self.ticket = ticket.resolve() if ticket else None
        self.credential = credential
        if self.ticket and (not credential or load(self.ticket)["token"] != credential):
            raise PermissionError("A matching one-shot run credential is required")

    def validate_model(self, identity: dict) -> None:
        """Bind any acceptance worker to exactly the claimed frozen model policy.

        Args:
            identity: Actual checkpoint/configuration/implementation fingerprint.
        """
        if self.ticket and load(self.ticket)["model_policy_sha256"] != digest(identity):
            raise PermissionError("Acceptance worker model differs from its claim")

    def output(self, path: Path) -> Path:
        """Keep runtime outputs and caches away from sealed media and private gold.

        Args:
            path: Requested output or cache directory.

        Returns:
            Authorized owned output path, restricted to the active worker when claimed.
        """
        actual = ensure_output_path(path)
        if self.ticket:
            claim = load(self.ticket)
            if not actual.is_relative_to(Path(claim["work_root"])):
                raise PermissionError("Acceptance output must stay inside its worker")
        self.check(actual / "runtime_output_guard")
        return actual

    def check(self, path: Path) -> Path:
        """Check a development input or an exact file authorized for one running job.

        Args:
            path: Input file, including manifests and images.

        Returns:
            Checked absolute path.

        Raises:
            PermissionError: The data are protected or outside the authorized run.
        """
        actual = path.resolve()
        if self.ticket is None:
            if actual in {
                PROJECT_ROOT / "configs/module2_selected_runtime.json",
                PROJECT_ROOT / "configs/module2_selected_development_candidate.json",
            }:
                return actual
            require_data_access([actual], "development", self.registry)
            return actual
        claim = load(self.ticket)
        if (
            claim["state"] != "running"
            or claim["token"] != self.credential
            or not actual.is_relative_to(PROJECT_ROOT)
        ):
            raise PermissionError("Acceptance ticket is not active")
        if actual.is_relative_to(Path(claim["private_gold_root"])):
            raise PermissionError("The model worker cannot read evaluator gold")
        if actual.is_relative_to(Path(claim["work_root"])):
            return actual
        if claim["authorized_files"].get(str(actual)) != sha256_file(actual):
            raise PermissionError("Input not bound by this acceptance claim")
        return actual


def public_records(path: Path, kind: str, access: RuntimeAccess) -> list[dict]:
    """Validate public catalog/query manifests and reject annotation fields.

    Args:
        path: A JSON array of catalog or query records.
        kind: Catalog or queries.
        access: Current development or scoped acceptance access policy.

    Returns:
        Canonical records with verified image hashes and absolute paths.

    Raises:
        ValueError: Public schema, image identity or unique record IDs are invalid.
    """
    path = access.check(path)
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    fields = {"id", "image_path", "image_sha256"} | (
        {"title"} if kind == "catalog" else {"mode", "text", "exclusions"}
    )
    if kind not in {"catalog", "queries"} or not isinstance(rows, list) or not rows:
        raise ValueError("A nonempty public catalog or query array is required")
    result, ids, images = [], set(), set()
    for raw in rows:
        if (
            not isinstance(raw, dict)
            or set(raw) - fields
            or not isinstance(raw.get("id"), str)
            or not raw["id"].strip()
            or raw["id"] in ids
        ):
            raise ValueError("Unique public IDs and no annotation fields required")
        ids.add(raw["id"])
        row = dict(raw)
        if kind == "catalog":
            if not isinstance(row.get("title"), str) or not row["title"].strip():
                raise ValueError("An original product title is required")
        else:
            mode = row.get("mode")
            if (
                mode not in ("image", "text")
                or (mode == "text") != bool(row.get("text"))
                or (mode == "image") != bool(row.get("image_path"))
            ):
                raise ValueError("Provide exactly one query modality")
            row.setdefault("exclusions", [])
            if not isinstance(row["exclusions"], list) or any(
                not isinstance(i, str) for i in row["exclusions"]
            ):
                raise ValueError("Explicit exclusions must be product IDs")
        if row.get("image_path"):
            image = Path(row["image_path"])
            if not image.is_absolute():
                image = PROJECT_ROOT / image
            image = access.check(image)
            actual = sha256_file(image)
            if row.get("image_sha256", actual) != actual:
                raise ValueError("Image identity differs from its manifest")
            with Image.open(image) as picture:
                picture.verify()
            row.update(image_path=str(image), image_sha256=actual)
            if kind == "catalog" and actual in images:
                raise ValueError("Duplicate catalog image bytes")
            images.add(actual)
        elif kind == "catalog":
            raise ValueError("A real catalog image is required")
        result.append(row)
    if kind == "catalog" and len(result) < 20:
        raise ValueError("At least twenty catalog products are required")
    return result


class ContentCache:
    """Cache model observations by full semantic input and model-policy identity."""

    def __init__(self, root: Path, identity: dict) -> None:
        """Create a content namespace without loading any cached predictions.

        Args:
            root: Owned cache directory.
            identity: Effective weights, prompts, preprocessing and policy hashes.
        """
        self.root = ensure_output_path(root)
        self.identity = identity
        self.hits = 0

    def key(self, stage: str, content: object) -> dict:
        """Build an observation key independent of arbitrary query/product IDs.

        Args:
            stage: Observation type.
            content: Public content fingerprints and required text.

        Returns:
            Full identity to serialize beside the cached observation.
        """
        return dict(stage=stage, model_policy=self.identity, content=content)

    def get(self, key: dict) -> object | None:
        """Read a matching observation, failing if stored content was altered.

        Args:
            key: Complete content/model identity.

        Returns:
            Cached JSON value or no observation.
        """
        path = self.root / (digest(key) + ".json")
        if not path.exists():
            return None
        row = load(path)
        if row["key"] != key or row["value_sha256"] != digest(row["value"]):
            raise ValueError("Cached observation was altered")
        self.hits += 1
        return row["value"]

    def put(self, key: dict, value: object) -> None:
        """Write an observation with its own content checksum.

        Args:
            key: Full content/model identity.
            value: Validated actual model output.
        """
        atomic_json(
            self.root / (digest(key) + ".json"),
            dict(key=key, value=value, value_sha256=digest(value)),
        )
