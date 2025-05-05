#!/usr/bin/env python3
"""
Sanitize_whitespace.py

A utility script to detect and fix whitespace-related issues in Python files.
Specifically targets:
- Non-printable characters (like U+00A0 non-breaking spaces)
- Inconsistent indentation
- Mixed tabs and spaces
- Trailing whitespace

This script runs before GitHub actions to prevent build failures.

Usage:
    python Sanitize_whitespace.py [directory_or_file]
    
    If no arguments provided, it scans all .py files in the current directory recursively.
"""

import os
import re
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Set, Dict, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("sanitize_whitespace")

# Non-printable characters to replace
PROBLEMATIC_CHARS = {
    '\u00A0': ' ',  # Non-breaking space -> Regular space
    '\u200B': '',    # Zero-width space -> Remove
    '\u2028': '\n',  # Line separator -> Newline
    '\u2029': '\n',  # Paragraph separator -> Newline
    '\u200D': '',    # Zero-width joiner -> Remove
    '\u202A': '',    # Left-to-right embedding -> Remove
    '\u202B': '',    # Right-to-left embedding -> Remove
    '\u202C': '',    # Pop directional formatting -> Remove
    '\u202D': '',    # Left-to-right override -> Remove
    '\u202E': '',    # Right-to-left override -> Remove
    '\t': '    '     # Tab -> 4 spaces (configurable)
}

# Files and directories to ignore
DEFAULT_IGNORE_PATTERNS = [
    r'\.git',
    r'\.github',
    r'__pycache__',
    r'\.pytest_cache',
    r'\.venv',
    r'venv',
    r'\.env',
    r'\.idea',
    r'\.vscode',
    r'\.mypy_cache',
    r'\.tox',
    r'\.coverage',
    r'htmlcov',
    r'dist',
    r'build',
    r'.*\.egg-info',
]


class WhitespaceSanitizer:
    """Class to handle the detection and fixing of whitespace issues."""
    
    def __init__(
        self, 
        target_path: Union[str, Path], 
        dry_run: bool = False,
        tab_size: int = 4,
        fix_indentation: bool = True,
        verbose: bool = False,
        ignore_patterns: Optional[List[str]] = None
    ):
        """
        Initialize the sanitizer.
        
        Args:
            target_path: Directory or file to process
            dry_run: If True, only report issues without fixing
            tab_size: Number of spaces to replace tabs with
            fix_indentation: Whether to attempt to fix indentation issues
            verbose: If True, show more detailed output
            ignore_patterns: List of regex patterns to ignore
        """
        self.target_path = Path(target_path)
        self.dry_run = dry_run
        self.tab_size = tab_size
        self.fix_indentation = fix_indentation
        self.verbose = verbose
        self.ignore_patterns = ignore_patterns or DEFAULT_IGNORE_PATTERNS
        
        # Update tab replacement based on tab_size
        PROBLEMATIC_CHARS['\t'] = ' ' * self.tab_size
        
        # Statistics
        self.files_processed = 0
        self.files_modified = 0
        self.chars_replaced: Dict[str, int] = {char: 0 for char in PROBLEMATIC_CHARS}
        self.indentation_fixes = 0
    
    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on ignore patterns."""
        path_str = str(path)
        return any(re.search(pattern, path_str) for pattern in self.ignore_patterns)

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the target path."""
        python_files = []
        
        if self.target_path.is_file() and self.target_path.suffix == '.py':
            return [self.target_path]
        
        for root, dirs, files in os.walk(self.target_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self.should_ignore(Path(root) / d)]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if not self.should_ignore(file_path):
                        python_files.append(file_path)
        
        return python_files

    def sanitize_file(self, file_path: Path) -> bool:
        """
        Sanitize a single Python file.
        
        Returns:
            bool: True if the file was modified, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning(f"Failed to read {file_path} with utf-8 encoding. Skipping.")
            return False
        
        original_content = content
        modified = False
        
        # Check for problematic characters
        has_problematic_chars = any(char in content for char in PROBLEMATIC_CHARS)
        
        # Replace problematic characters
        for char, replacement in PROBLEMATIC_CHARS.items():
            if char in content:
                count = content.count(char)
                if count > 0:
                    content = content.replace(char, replacement)
                    self.chars_replaced[char] += count
                    modified = True
                    if self.verbose:
                        logger.info(f"Replaced {count} occurrences of {repr(char)} in {file_path}")
        
        # Fix common indentation issues if requested
        if self.fix_indentation:
            fixed_content, indentation_fixes = self.fix_indentation_issues(content)
            if indentation_fixes > 0:
                content = fixed_content
                self.indentation_fixes += indentation_fixes
                modified = True
                if self.verbose:
                    logger.info(f"Fixed {indentation_fixes} indentation issues in {file_path}")
        
        # Remove trailing whitespace
        new_content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
        if new_content != content:
            content = new_content
            modified = True
            if self.verbose:
                logger.info(f"Removed trailing whitespace in {file_path}")
        
        # Write the changes back to the file if modified and not in dry run mode
        if modified and not self.dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return modified

    def fix_indentation_issues(self, content: str) -> Tuple[str, int]:
        """
        Attempt to fix common indentation issues.
        
        Returns:
            Tuple[str, int]: (Fixed content, Number of fixes made)
        """
        lines = content.split('\n')
        fixes = 0
        
        # Check for mixed indentation within a block
        indentation_levels = []
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('#'):
                fixed_lines.append(line)
                continue
                
            # Get the indentation level
            match = re.match(r'^(\s*)', line)
            if not match:
                fixed_lines.append(line)
                continue
                
            indentation = match.group(1)
            
            # Handle mixed spaces in indentation
            if ' ' in indentation and indentation.count(' ') % self.tab_size != 0:
                # Normalize to multiples of tab_size
                spaces_count = len(indentation)
                normalized_spaces = (spaces_count // self.tab_size) * self.tab_size
                if normalized_spaces != spaces_count:
                    # Check if we need to add or remove spaces
                    if spaces_count - normalized_spaces < self.tab_size / 2:
                        # Remove spaces to go to lower indentation level
                        new_indentation = ' ' * normalized_spaces
                    else:
                        # Add spaces to go to higher indentation level
                        new_indentation = ' ' * (normalized_spaces + self.tab_size)
                    
                    fixed_line = new_indentation + line.lstrip()
                    fixed_lines.append(fixed_line)
                    fixes += 1
                    continue
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines), fixes

    def run(self) -> Dict:
        """
        Run the sanitization process.
        
        Returns:
            Dict: Statistics about the sanitization
        """
        if not self.target_path.exists():
            logger.error(f"Path not found: {self.target_path}")
            return {
                "success": False,
                "error": f"Path not found: {self.target_path}"
            }
        
        python_files = self.find_python_files()
        logger.info(f"Found {len(python_files)} Python files to check")
        
        for file_path in python_files:
            self.files_processed += 1
            modified = self.sanitize_file(file_path)
            if modified:
                self.files_modified += 1
                logger.info(f"{'Would fix' if self.dry_run else 'Fixed'} whitespace issues in {file_path}")
        
        # Report statistics
        logger.info(f"Processed {self.files_processed} files")
        logger.info(f"{'Would modify' if self.dry_run else 'Modified'} {self.files_modified} files")
        
        for char, count in self.chars_replaced.items():
            if count > 0:
                logger.info(f"Replaced {count} occurrences of {repr(char)}")
        
        if self.indentation_fixes > 0:
            logger.info(f"Fixed {self.indentation_fixes} indentation issues")
        
        return {
            "success": True,
            "files_processed": self.files_processed,
            "files_modified": self.files_modified,
            "chars_replaced": self.chars_replaced,
            "indentation_fixes": self.indentation_fixes
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect and fix whitespace-related issues in Python files."
    )
    parser.add_argument(
        "path", 
        nargs="?", 
        default=".",
        help="Directory or file to process (default: current directory)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Report issues without fixing them"
    )
    parser.add_argument(
        "--tab-size", 
        type=int, 
        default=4,
        help="Number of spaces to replace tabs with (default: 4)"
    )
    parser.add_argument(
        "--no-fix-indentation", 
        action="store_false", 
        dest="fix_indentation",
        help="Don't attempt to fix indentation issues"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show more detailed output"
    )
    parser.add_argument(
        "--ignore", 
        type=str, 
        nargs="+",
        help="Additional patterns to ignore (regex)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    ignore_patterns = DEFAULT_IGNORE_PATTERNS
    if args.ignore:
        ignore_patterns.extend(args.ignore)
    
    sanitizer = WhitespaceSanitizer(
        target_path=args.path,
        dry_run=args.dry_run,
        tab_size=args.tab_size,
        fix_indentation=args.fix_indentation,
        verbose=args.verbose,
        ignore_patterns=ignore_patterns
    )
    
    result = sanitizer.run()
    
    if not result["success"]:
        sys.exit(1)
    
    # Exit with status code 1 if issues were found and fixed
    # This helps in CI pipelines to indicate that changes were made
    if result["files_modified"] > 0 and not args.dry_run:
        logger.warning("Whitespace issues were found and fixed. Please commit the changes.")
        sys.exit(1)
    elif result["files_modified"] > 0 and args.dry_run:
        logger.warning("Whitespace issues were found. Run without --dry-run to fix them.")
        sys.exit(1)
    else:
        logger.info("No whitespace issues found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
