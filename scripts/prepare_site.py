#!/usr/bin/env python3
"""Stage repository notes for MkDocs without moving their source files."""

from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / ".site-content"
SITE_FILES = ("index.md", "about.md")
SITE_DIRECTORIES = ("assets", "javascripts", "pictures", "stylesheets")
# Repository instructions and site-maintenance pages are not articles.
EXCLUDED_MARKDOWN = {*SITE_FILES, "README.md", "design-qa.md", "archive.md"}


def discover_documents() -> list[Path]:
    # Only tracked files belong to the public archive; local drafts stay private.
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.md"], cwd=ROOT
    ).decode("utf-8")
    return sorted(
        Path(name)
        for name in tracked.split("\0")
        if name
        and name not in EXCLUDED_MARKDOWN
        and Path(name).name not in {"AGENTS.md", "AGENTS.local.md"}
    )


def document_title(path: Path) -> str:
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("_", " ").replace("-", " ")


def write_archive(documents: list[Path]) -> None:
    sections = (
        ("Articles", [path for path in documents if path.parent == Path(".")]),
        ("Supporting documents", [path for path in documents if path.parent != Path(".")]),
    )
    lines = [
        "---", "title: Archive", "hide:", "  - actions", "---", "",
        "# Archive", "",
        f"Browse all {len(documents)} notes and supporting documents, listed alphabetically by title.",
        "",
    ]
    for heading, paths in sections:
        if not paths:
            continue
        lines.extend([f"## {heading} ({len(paths)})", ""])
        entries = [(document_title(path), path) for path in paths]
        for title, path in sorted(entries, key=lambda entry: (entry[0].casefold(), entry[1])):
            label = title.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
            lines.append(f"- [{label}](<{quote(path.as_posix(), safe='/')}>)")
        lines.append("")
    (STAGE / "archive.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    documents = discover_documents()
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()

    for filename in SITE_FILES:
        source = ROOT / filename
        shutil.copy2(source, STAGE / filename)

    for dirname in SITE_DIRECTORIES:
        source = ROOT / dirname
        # Markdown is staged separately so untracked drafts are never published.
        shutil.copytree(source, STAGE / dirname, ignore=shutil.ignore_patterns("*.md"))

    for path in documents:
        destination = STAGE / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, destination)

    write_archive(documents)


if __name__ == "__main__":
    main()
