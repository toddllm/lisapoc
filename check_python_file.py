#!/usr/bin/env python3
"""
Script to check Python files for syntax errors and linting issues.
Usage: python check_python_file.py <filename>
"""

import sys
import os
import ast
import importlib.util
import subprocess
import tempfile

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def check_syntax(filename):
    """Check basic Python syntax using ast module."""
    print(f"{Colors.HEADER}Checking syntax with Python's ast module...{Colors.ENDC}")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        print(f"{Colors.GREEN}✓ No syntax errors detected with ast.{Colors.ENDC}")
        return True
    except SyntaxError as e:
        print(f"{Colors.RED}✗ Syntax error at line {e.lineno}, column {e.offset}: {e.msg}{Colors.ENDC}")
        # Show the offending line and position
        if hasattr(e, 'text') and e.text:
            print(f"Line {e.lineno}: {e.text.rstrip()}")
            if e.offset:
                print(f"{' ' * (e.offset + 10)}^ Here")
        return False

def check_compile(filename):
    """Check if the file can be compiled."""
    print(f"{Colors.HEADER}Trying to compile the file...{Colors.ENDC}")
    try:
        subprocess.run([sys.executable, '-m', 'py_compile', filename], 
                      check=True, capture_output=True, text=True)
        print(f"{Colors.GREEN}✓ File compiled successfully.{Colors.ENDC}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}✗ Compilation failed:{Colors.ENDC}")
        print(e.stderr)
        return False

def check_flake8(filename):
    """Check the file with flake8 if available."""
    if importlib.util.find_spec("flake8"):
        print(f"{Colors.HEADER}Checking with flake8...{Colors.ENDC}")
        try:
            result = subprocess.run(['flake8', filename], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"{Colors.GREEN}✓ No issues found by flake8.{Colors.ENDC}")
                return True
            else:
                print(f"{Colors.YELLOW}! Flake8 found issues:{Colors.ENDC}")
                print(result.stdout)
                return False
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}Error running flake8: {e}{Colors.ENDC}")
            return False
    else:
        print(f"{Colors.YELLOW}Flake8 not installed. Install with: pip install flake8{Colors.ENDC}")
        return None

def check_pylint(filename):
    """Check the file with pylint if available."""
    if importlib.util.find_spec("pylint"):
        print(f"{Colors.HEADER}Checking with pylint...{Colors.ENDC}")
        try:
            result = subprocess.run(['pylint', filename], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"{Colors.GREEN}✓ No issues found by pylint.{Colors.ENDC}")
                return True
            else:
                print(f"{Colors.YELLOW}! Pylint found issues:{Colors.ENDC}")
                # Only show first few lines to avoid overwhelming output
                lines = result.stdout.split('\n')
                for line in lines[:20]:  # Show first 20 lines
                    print(line)
                if len(lines) > 20:
                    print(f"... and {len(lines) - 20} more lines.")
                return False
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}Error running pylint: {e}{Colors.ENDC}")
            return False
    else:
        print(f"{Colors.YELLOW}Pylint not installed. Install with: pip install pylint{Colors.ENDC}")
        return None

def check_indent_errors(filename):
    """Check for indentation errors specifically."""
    print(f"{Colors.HEADER}Checking for indentation errors...{Colors.ENDC}")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.readlines()
        
        errors = []
        stack = [0]  # Stack to keep track of indentation levels
        
        for i, line in enumerate(content, 1):
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
                
            # Calculate indentation
            indent = len(line) - len(line.lstrip())
            
            # Check if indentation matches current level or introduces a new level
            if indent > stack[-1]:
                stack.append(indent)
            elif indent < stack[-1]:
                while stack and indent < stack[-1]:
                    stack.pop()
                if indent not in stack:
                    errors.append((i, line.rstrip(), "Inconsistent indentation"))
            
            # Check for specific control flow keywords that should have blocks after them
            if stripped.startswith(('if ', 'elif ', 'else:', 'for ', 'while ', 'def ', 'class ', 'try:', 'except ')):
                if not stripped.endswith(':'):
                    if not any(stripped.startswith(x) for x in ['if ', 'elif ', 'for ', 'while ', 'def ', 'class ']):
                        errors.append((i, line.rstrip(), "Missing colon"))
                
                # Check the next line for indentation if this is a block start
                if stripped.endswith(':') and i < len(content):
                    next_line = content[i].strip()
                    next_indent = len(content[i]) - len(content[i].lstrip())
                    
                    # Skip empty lines and comments when looking for the next content line
                    j = i
                    while j < len(content) - 1 and (not next_line or next_line.startswith('#')):
                        j += 1
                        next_line = content[j].strip()
                        next_indent = len(content[j]) - len(content[j].lstrip())
                    
                    if next_line and next_indent <= indent:
                        errors.append((i, line.rstrip(), "Expected an indented block"))
        
        if errors:
            print(f"{Colors.RED}✗ Found {len(errors)} potential indentation errors:{Colors.ENDC}")
            for line_num, line_text, error_msg in errors:
                print(f"Line {line_num}: {error_msg}")
                print(f"  {line_text}")
            return False
        else:
            print(f"{Colors.GREEN}✓ No obvious indentation errors detected.{Colors.ENDC}")
            return True
    except Exception as e:
        print(f"{Colors.RED}Error checking indentation: {e}{Colors.ENDC}")
        return False

def check_try_except_blocks(filename):
    """Check that every try block has a corresponding except or finally block."""
    print(f"{Colors.HEADER}Checking try-except block structures...{Colors.ENDC}")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.readlines()
        
        errors = []
        in_try_block = False
        try_line = 0
        
        for i, line in enumerate(content, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Check for try statements
            if stripped.startswith('try:'):
                in_try_block = True
                try_line = i
            
            # Check for except or finally
            elif in_try_block and (stripped.startswith('except ') or stripped.startswith('except:') or stripped.startswith('finally:')):
                in_try_block = False
            
            # If we find a line that's not indented more than the try and not an except/finally, it's an error
            elif in_try_block:
                try_indent = len(content[try_line-1]) - len(content[try_line-1].lstrip())
                current_indent = len(line) - len(line.lstrip())
                
                # If back to same or less indentation without except/finally, it's an error
                if current_indent <= try_indent and not stripped.startswith(('except', 'finally')):
                    errors.append((try_line, content[try_line-1].rstrip(), "Try block without except or finally"))
                    in_try_block = False
        
        if errors:
            print(f"{Colors.RED}✗ Found {len(errors)} incomplete try-except blocks:{Colors.ENDC}")
            for line_num, line_text, error_msg in errors:
                print(f"Line {line_num}: {error_msg}")
                print(f"  {line_text}")
            return False
        else:
            print(f"{Colors.GREEN}✓ All try blocks have corresponding except or finally clauses.{Colors.ENDC}")
            return True
    except Exception as e:
        print(f"{Colors.RED}Error checking try-except blocks: {e}{Colors.ENDC}")
        return False

def extract_problematic_section(filename, error_line, context=5):
    """Extract the section around the error for detailed analysis."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        start = max(0, error_line - context - 1)
        end = min(len(lines), error_line + context)
        
        print(f"{Colors.HEADER}Code section around line {error_line}:{Colors.ENDC}")
        for i, line in enumerate(lines[start:end], start + 1):
            prefix = "→ " if i == error_line else "  "
            print(f"{Colors.BOLD if i == error_line else ''}{prefix}{i}: {line.rstrip()}{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}Error extracting code section: {e}{Colors.ENDC}")

def fix_common_issues(filename):
    """Attempt to fix common syntax issues and save to a new file."""
    print(f"{Colors.HEADER}Attempting to fix common issues...{Colors.ENDC}")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        fixed_lines = []
        changes = 0
        
        # Check for and fix common issues
        for i, line in enumerate(lines):
            fixed_line = line
            
            # Fix missing colons in control structures
            if any(keyword in line.strip() for keyword in ['if ', 'elif ', 'else', 'for ', 'while ', 'def ', 'class ', 'try', 'except ']):
                if not line.strip().endswith(':') and not line.strip().endswith(':)'):
                    fixed_line = line.rstrip() + ':\n'
                    changes += 1
            
            # Ensure indentation is consistent (using 4 spaces)
            indent_count = len(line) - len(line.lstrip())
            if indent_count % 4 != 0 and line.strip():
                spaces_needed = ((indent_count // 4) + 1) * 4
                fixed_line = ' ' * spaces_needed + line.lstrip()
                changes += 1
            
            fixed_lines.append(fixed_line)
        
        # Write fixed content to a new file
        if changes > 0:
            fixed_filename = f"{os.path.splitext(filename)[0]}_fixed{os.path.splitext(filename)[1]}"
            with open(fixed_filename, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            print(f"{Colors.GREEN}Made {changes} fixes. Fixed file saved as: {fixed_filename}{Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}No automatic fixes were applied.{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}Error attempting to fix issues: {e}{Colors.ENDC}")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        sys.exit(1)
    
    print(f"{Colors.BOLD}{Colors.HEADER}Checking Python file: {filename}{Colors.ENDC}")
    print("-" * 60)
    
    # Run all checks
    syntax_ok = check_syntax(filename)
    indent_ok = check_indent_errors(filename)
    try_except_ok = check_try_except_blocks(filename)
    compile_ok = check_compile(filename)
    
    # Only run linters if syntax is okay
    if syntax_ok:
        flake8_ok = check_flake8(filename)
        pylint_ok = check_pylint(filename)
    
    # Print summary
    print("\n" + "-" * 60)
    print(f"{Colors.BOLD}{Colors.HEADER}Summary:{Colors.ENDC}")
    print(f"Syntax check: {'✓' if syntax_ok else '✗'}")
    print(f"Indentation: {'✓' if indent_ok else '✗'}")
    print(f"Try-except blocks: {'✓' if try_except_ok else '✗'}")
    print(f"Compilation: {'✓' if compile_ok else '✗'}")
    
    # Suggest fixes if there are issues
    if not (syntax_ok and indent_ok and try_except_ok and compile_ok):
        print("\n" + "-" * 60)
        print(f"{Colors.BOLD}{Colors.YELLOW}Suggestions for fixing errors:{Colors.ENDC}")
        print("1. Check for missing colons after if/else/for/while/try/except statements")
        print("2. Ensure every 'try:' has a corresponding 'except:' or 'finally:' block")
        print("3. Check for proper indentation (use 4 spaces per level)")
        print("4. Look for unbalanced parentheses, brackets, or braces")
        
        # Offer to attempt automatic fixes
        fix_common_issues(filename)

if __name__ == "__main__":
    main() 