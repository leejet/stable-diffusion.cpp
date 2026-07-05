#!/usr/bin/env python3
"""Remove UTF-8 BOMs from files under a directory.

By default this scans the current working directory recursively and skips
repository areas that should not be touched by ordinary maintenance scripts.
Only files whose first three bytes are the UTF-8 BOM are rewritten.
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


UTF8_BOM = b"\xef\xbb\xbf"

DEFAULT_EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "test",
}

DEFAULT_EXCLUDED_DIR_PREFIXES = {
    "build",
}

DEFAULT_EXCLUDED_REL_DIRS = {
    "examples/server/frontend",
    "ggml",
    "models",
    "src/vocab",
    "thirdparty",
}


def rel_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def should_skip_dir(
    path: Path,
    root: Path,
    excluded_rel_dirs: set[str],
    excluded_names: set[str],
    excluded_prefixes: set[str],
) -> bool:
    rel = rel_posix(path, root)
    return (
        path.name in excluded_names
        or rel in excluded_rel_dirs
        or any(path.name.startswith(prefix) for prefix in excluded_prefixes)
    )


def iter_files(
    root: Path,
    recursive: bool,
    excluded_rel_dirs: set[str],
    excluded_names: set[str],
    excluded_prefixes: set[str],
    follow_symlinks: bool,
):
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
            current_dir = Path(dirpath)
            dirnames[:] = [
                name
                for name in dirnames
                if not should_skip_dir(
                    current_dir / name,
                    root,
                    excluded_rel_dirs,
                    excluded_names,
                    excluded_prefixes,
                )
            ]
            for filename in filenames:
                path = current_dir / filename
                if path.is_symlink() and not follow_symlinks:
                    continue
                yield path
    else:
        for path in root.iterdir():
            if path.is_file() and (follow_symlinks or not path.is_symlink()):
                yield path


def has_utf8_bom(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(len(UTF8_BOM)) == UTF8_BOM


def strip_utf8_bom(path: Path) -> None:
    tmp_path = None
    try:
        with path.open("rb") as src:
            if src.read(len(UTF8_BOM)) != UTF8_BOM:
                return

            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=str(path.parent),
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)

        shutil.copystat(path, tmp_path, follow_symlinks=False)
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan files and convert UTF-8 BOM files to UTF-8 without BOM.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Directory to scan. Defaults to the current directory.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Only list files that would be converted.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan files directly under root.",
    )
    parser.add_argument(
        "--include-repo-excluded",
        action="store_true",
        help="Do not skip default repository excluded directories.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Additional directory name or root-relative path to skip. Can be used multiple times.",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symlinked directories and files.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print the final summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()

    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    excluded_names = set()
    excluded_rel_dirs = set()
    excluded_prefixes = set()
    if not args.include_repo_excluded:
        excluded_names.update(DEFAULT_EXCLUDED_DIR_NAMES)
        excluded_rel_dirs.update(DEFAULT_EXCLUDED_REL_DIRS)
        excluded_prefixes.update(DEFAULT_EXCLUDED_DIR_PREFIXES)

    for item in args.exclude_dir:
        normalized = Path(item).as_posix().strip("/")
        if "/" in normalized:
            excluded_rel_dirs.add(normalized)
        else:
            excluded_names.add(normalized)

    scanned = 0
    converted = 0
    errors = 0

    for path in iter_files(
        root=root,
        recursive=not args.no_recursive,
        excluded_rel_dirs=excluded_rel_dirs,
        excluded_names=excluded_names,
        excluded_prefixes=excluded_prefixes,
        follow_symlinks=args.follow_symlinks,
    ):
        scanned += 1
        try:
            if not has_utf8_bom(path):
                continue
            converted += 1
            rel = rel_posix(path, root)
            if args.dry_run:
                if not args.quiet:
                    print(f"would convert: {rel}")
            else:
                strip_utf8_bom(path)
                if not args.quiet:
                    print(f"converted: {rel}")
        except OSError as exc:
            errors += 1
            print(f"error: {rel_posix(path, root)}: {exc}", file=sys.stderr)

    action = "would convert" if args.dry_run else "converted"
    print(f"scanned {scanned} file(s), {action} {converted}, errors {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
