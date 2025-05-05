#!/usr/bin/env python3

import sys
import pathlib

def sanitize_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        # Replace non-breaking spaces (U+00A0) with normal spaces
        line = line.replace('\u00A0', ' ')
        # Replace tabs with four spaces (optional, for consistent indentation)
        line = line.replace('\t', '    ')
        cleaned_lines.append(line.rstrip() + '\n')  # Trim trailing spaces

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"Sanitized: {file_path}")

def sanitize_directory(target_dir):
    py_files = pathlib.Path(target_dir).rglob("*.py")
    for py_file in py_files:
        sanitize_file(py_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sanitize_whitespace.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    sanitize_directory(directory)
    print("Sanitization complete.")
