#!/usr/bin/env python3
"""Apply TRL #6361 fix: Add elif branch for think-end-only content in Qwen3.5 templates.

This script modifies the jinja template files directly on the server where TRL is installed.
It finds the think-detection block and inserts an elif branch to handle the case where
the generation prompt emitted <think>, but the content only has </think>.

Usage:
    python3 apply_6361_fix.py [--dry-run] [--env-path PATH]

Default env-path: /jiangdingfeng/miniconda3/envs/env-flex
"""

import os
import sys
import shutil

THINK_START = "<think>"
THINK_END = "</think>"

# Default TRL install path on the partner server
DEFAULT_ENV_PATH = "/jiangdingfeng/miniconda3/envs/env-flex"

TEMPLATE_NAMES = [
    "qwen3_5_think_training.jinja",
    "qwen3_5_nothink_training.jinja",
]

# The elif branch lines to insert before the closing {%- endif %}
ELIF_LINES = [
    "            {%- elif '</think>' in content %}",
    "                {#- <think> was from generation prompt; </think> in content closes it #}",
    "                {%- set reasoning_content = content.split('</think>')[0].lstrip('\\n') %}",
    "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}",
]


def find_template_dir(env_path):
    """Find the TRL chat_templates directory within a conda env."""
    # Try common python versions
    for py_ver in ["python3.12", "python3.11", "python3.10", "python3.9"]:
        site_packages = os.path.join(env_path, "lib", py_ver, "site-packages")
        templates_dir = os.path.join(site_packages, "trl", "chat_templates")
        if os.path.exists(templates_dir):
            return templates_dir, os.path.join(site_packages, "trl")

    print(f"ERROR: No chat_templates dir found under {env_path}")
    return None, None


def apply_fix(template_content, dry_run=False):
    """Add elif branch for think-end-only content in the template.

    Finds the pattern:
        {%- if '<think>' in content and '</think>' in content %}
            {%- set reasoning_content = ... %}
            {%- set content = ... %}
        {%- endif %}

    And changes it to:
        {%- if '<think>' in content and '</think>' in content %}
            {%- set reasoning_content = ... %}
            {%- set content = ... %}
        {%- elif '</think>' in content %}
            {#- <think> was from generation prompt; </think> closes it #}
            {%- set reasoning_content = content.split('</think>')[0].lstrip('\n') %}
            {%- set content = content.split('</think>')[-1].lstrip('\n') %}
        {%- endif %}
    """
    lines = template_content.split('\n')
    fixed_lines = []
    fix_applied = False

    # Find the think-detection if block: {%- if '<think>' in content and '</think>' in content %}
    think_if_line = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Only match actual Jinja2 if statements (not comment/diff lines starting with -)
        if stripped.startswith("{%- if") and THINK_START in stripped and THINK_END in stripped and 'in content' in stripped:
            think_if_line = i
            print("  Found think-detection if at L%d: %s" % (i+1, stripped[:80]))
            break

    if think_if_line is None:
        print("  WARNING: No think-detection if block found.")
        return template_content, False

    # Find the closing endif for this inner if block
    # Structure: if -> set -> set -> endif (inner) -> endif (outer)
    endif_line = None
    for i in range(think_if_line + 1, min(think_if_line + 10, len(lines))):
        stripped = lines[i].strip()
        if stripped == '{%- endif %}':
            endif_line = i
            print("  Found closing endif at L%d" % (i+1))
            break

    if endif_line is None:
        print("  WARNING: Could not find closing endif for think-detection block")
        return template_content, False

    # Insert the elif lines before the endif
    for i, line in enumerate(lines):
        if i == endif_line:
            for elif_line in ELIF_LINES:
                fixed_lines.append(elif_line)
            fix_applied = True
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    if fix_applied:
        status = "DRY RUN: Would apply" if dry_run else "Applied"
        print("  %s: elif branch inserted before endif at L%d" % (status, endif_line+1))

    return '\n'.join(fixed_lines), fix_applied


def verify_fix(template_content):
    """Verify the elif branch exists after the if branch."""
    lines = template_content.split('\n')

    has_if = False
    has_elif = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("{%- if") and THINK_START in stripped and THINK_END in stripped and 'in content' in stripped:
            has_if = True
        if "elif '</think>'" in stripped and 'in content' in stripped:
            has_elif = True

    if has_if and has_elif:
        print("  VERIFIED: Both if and elif branches present")
        return True
    elif has_if and not has_elif:
        print("  VERIFICATION FAILED: if present but no elif (bug still exists)")
        return False
    else:
        print("  VERIFICATION: No think-detection if found")
        return False


def show_diff(original, fixed):
    """Show a simple diff of the changes."""
    orig_lines = original.split('\n')
    fix_lines = fixed.split('\n')
    print("\n  === Diff ===")
    for i in range(max(len(orig_lines), len(fix_lines))):
        if i >= len(orig_lines):
            print("  + L%d: %s" % (i+1, fix_lines[i]))
        elif i >= len(fix_lines):
            print("  - L%d: %s" % (i+1, orig_lines[i]))
        elif orig_lines[i] != fix_lines[i]:
            print("  - L%d: %s" % (i+1, orig_lines[i]))
            print("  + L%d: %s" % (i+1, fix_lines[i]))


def main():
    dry_run = "--dry-run" in sys.argv
    env_path = DEFAULT_ENV_PATH

    # Check for custom env path
    for i, arg in enumerate(sys.argv):
        if arg == "--env-path" and i + 1 < len(sys.argv):
            env_path = sys.argv[i + 1]

    templates_dir, trl_path = find_template_dir(env_path)
    if not templates_dir:
        # Try env-rollout as fallback
        templates_dir, trl_path = find_template_dir("/jiangdingfeng/miniconda3/envs/env-rollout")
        if templates_dir:
            print("Found TRL templates at: %s" % templates_dir)
        else:
            print("ERROR: Could not find TRL templates directory")
            sys.exit(1)
    else:
        print("Found TRL templates at: %s" % templates_dir)

    # Check TRL version
    init_file = os.path.join(trl_path, "__init__.py")
    if os.path.exists(init_file):
        with open(init_file) as f:
            for line in f:
                if "__version__" in line:
                    print("TRL version: %s" % line.strip())
                    break

    all_ok = True

    for template_name in TEMPLATE_NAMES:
        template_path = os.path.join(templates_dir, template_name)

        if not os.path.exists(template_path):
            print("\n=== %s NOT FOUND ===" % template_name)
            continue

        print("\n=== %s ===" % template_name)

        with open(template_path) as f:
            original_content = f.read()

        print("  Original size: %d chars" % len(original_content))

        # Show the bug lines
        orig_lines = original_content.split('\n')
        for i, line in enumerate(orig_lines):
            stripped = line.strip()
            if stripped.startswith("{%- if") and THINK_START in stripped and THINK_END in stripped:
                print("  Bug if at L%d: %s" % (i+1, stripped))
                # Show next few lines
                for j in range(i+1, min(i+5, len(orig_lines))):
                    print("  L%d: %s" % (j+1, orig_lines[j].strip()))

        # Check if fix is already applied
        if verify_fix(original_content):
            print("  Fix already applied, skipping")
            continue

        # Apply the fix
        fixed_content, fix_applied = apply_fix(original_content, dry_run=dry_run)

        if fix_applied:
            show_diff(original_content, fixed_content)

            if not dry_run:
                # Backup original
                backup_path = template_path + ".orig"
                if not os.path.exists(backup_path):
                    shutil.copy2(template_path, backup_path)
                    print("  Backup saved to: %s" % backup_path)

                # Write the fixed template
                with open(template_path, 'w') as f:
                    f.write(fixed_content)
                print("  Fixed template written: %d chars" % len(fixed_content))

                # Verify the fix
                with open(template_path) as f:
                    verify_content = f.read()
                if not verify_fix(verify_content):
                    print("  ERROR: Fix verification failed!")
                    shutil.copy2(backup_path, template_path)
                    print("  Original restored from backup")
                    all_ok = False
                else:
                    print("  Fix successfully verified on disk")

    if all_ok:
        print("\n=== All fixes applied successfully ===")
    else:
        print("\n=== Some fixes failed ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
