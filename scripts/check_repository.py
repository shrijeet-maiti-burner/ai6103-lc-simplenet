"""Check local Markdown links and preserved-source hashes without network access."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    errors = []
    paths = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "THIRD_PARTY.md"]
    paths += list((ROOT / "docs").rglob("*.md"))
    paths += [ROOT / p / "README.md" for p in ("data", "results", "report")]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if text.count("```") % 2:
            errors.append(f"Unclosed fence: {path.relative_to(ROOT)}")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            target = target.split("#", 1)[0]
            if not (path.parent / target).exists():
                errors.append(f"Missing link: {path.relative_to(ROOT)} -> {target}")
    manifest = json.loads((ROOT / "references/source-manifest.json").read_text(encoding="utf-8"))
    for item in manifest["preserved"]:
        path = ROOT / item["repo_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"Changed preserved source: {item['repo_path']}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(paths)} Markdown files and {len(manifest['preserved'])} source hashes.")


if __name__ == "__main__":
    main()
