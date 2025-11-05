#!/usr/bin/env python3
"""
Script to convert argparse arguments from OpenAI Whisper's transcribe.py
to CLAMS AppMetadata runtime parameters.

Usage:
    python convert_whisper_args.py <github_url>

Example:
    python convert_whisper_args.py https://github.com/openai/whisper/blob/c0d2f624c09dc18e709e37c2ad90c039a4eb72a2/whisper/transcribe.py
"""

import argparse
import re
import sys
import urllib.request


def fetch_file_content(url):
    """Fetch content from a GitHub URL (blob or raw)."""
    # Convert blob URL to raw URL
    raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')

    try:
        with urllib.request.urlopen(raw_url) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        print(f"Error fetching URL: {e}", file=sys.stderr)
        sys.exit(1)


def extract_argparse_section(content):
    """Extract argparse argument definitions from the file content."""
    lines = content.split('\n')

    # Find where the parser is created
    parser_start = None
    for i, line in enumerate(lines):
        if 'argparse.ArgumentParser' in line:
            parser_start = i
            break

    if parser_start is None:
        print("Error: Could not find ArgumentParser creation", file=sys.stderr)
        sys.exit(1)

    # Collect all parser.add_argument() calls
    arguments = []
    current_arg = []
    in_argument = False
    paren_depth = 0

    for i in range(parser_start, len(lines)):
        line = lines[i]

        if 'parser.add_argument' in line:
            if current_arg:  # Save previous argument
                arguments.append('\n'.join(current_arg))
            current_arg = [line]
            in_argument = True
            paren_depth = line.count('(') - line.count(')')
        elif in_argument:
            current_arg.append(line)
            paren_depth += line.count('(') - line.count(')')

            if paren_depth == 0:
                arguments.append('\n'.join(current_arg))
                current_arg = []
                in_argument = False

        # Stop at the end of the function or at args = parser.parse_args()
        if 'parse_args()' in line and not in_argument:
            break

    return arguments


def parse_argument(arg_text):
    """Parse a single argparse argument definition."""
    # Extract argument name (first positional argument)
    name_match = re.search(r'add_argument\(["\']([^"\']+)["\']', arg_text)
    if not name_match:
        # Try positional without quotes or with parentheses
        name_match = re.search(r'add_argument\("([^"]+)"', arg_text)

    if not name_match:
        return None

    name = name_match.group(1).lstrip('-')

    # Extract parameters
    arg_info = {
        'name': name,
        'original_name': name_match.group(1)
    }

    # Extract type
    type_match = re.search(r'type\s*=\s*([^,\)]+)', arg_text)
    if type_match:
        type_str = type_match.group(1).strip()
        arg_info['type'] = type_str

    # Extract default
    default_match = re.search(r'default\s*=\s*([^,\)]+)', arg_text)
    if default_match:
        default_str = default_match.group(1).strip()
        arg_info['default'] = default_str

    # Extract choices
    choices_match = re.search(r'choices\s*=\s*\[([^\]]+)\]', arg_text)
    if choices_match:
        arg_info['choices'] = choices_match.group(1)

    # Extract help text
    help_match = re.search(r'help\s*=\s*["\']([^"\']*(?:["\'][^"\']*)*)["\']', arg_text, re.DOTALL)
    if help_match:
        arg_info['help'] = help_match.group(1).strip()

    # Extract nargs
    nargs_match = re.search(r'nargs\s*=\s*["\']([^"\']+)["\']', arg_text)
    if nargs_match:
        arg_info['nargs'] = nargs_match.group(1)

    return arg_info


def map_type_to_clams(whisper_type):
    """Map Whisper argparse types to CLAMS parameter types."""
    type_mapping = {
        'str': 'string',
        'int': 'integer',
        'float': 'number',
        'str2bool': 'boolean',
        'optional_int': 'integer',
        'optional_float': 'number',
        'valid_model_name': 'string',
    }

    for key, value in type_mapping.items():
        if key in whisper_type:
            return value

    return 'string'  # default


def format_default_value(default_str, param_type):
    """Format default value for CLAMS metadata."""
    if default_str == 'None':
        if param_type == 'string':
            return "''"
        elif param_type == 'integer':
            return '0'
        elif param_type == 'number':
            return '0.0'
        elif param_type == 'boolean':
            return 'False'

    # Handle boolean strings
    if default_str in ['True', 'False']:
        return default_str

    # Handle numeric values
    if param_type in ['integer', 'number']:
        return default_str

    # Handle string values - ensure they're quoted
    if param_type == 'string' and not (default_str.startswith('"') or default_str.startswith("'")):
        return f'"{default_str}"'

    return default_str


def convert_to_clams_parameter(arg_info, prefix="(from openai-whisper CLI) "):
    """Convert parsed argument info to CLAMS metadata.add_parameter() call."""
    if not arg_info or arg_info.get('original_name', '').startswith('audio'):
        return None  # Skip positional arguments like 'audio'

    # Convert snake_case to camelCase for parameter name
    parts = arg_info['name'].split('_')
    camel_name = parts[0] + ''.join(word.capitalize() for word in parts[1:])

    # Determine parameter type
    param_type = 'string'
    if 'type' in arg_info:
        param_type = map_type_to_clams(arg_info['type'])

    # Build the parameter definition
    param_lines = [f"    metadata.add_parameter("]
    param_lines.append(f'        name="{camel_name}",')
    param_lines.append(f"        type='{param_type}',")

    # Add default value
    if 'default' in arg_info:
        default_val = format_default_value(arg_info['default'], param_type)
        param_lines.append(f"        default={default_val},")

    # Add choices if present
    if 'choices' in arg_info:
        choices_str = arg_info['choices']
        # Clean up the choices string
        choices_str = re.sub(r'\s+', ' ', choices_str)
        param_lines.append(f"        choices=[{choices_str}],")

    # Add description (help text)
    if 'help' in arg_info:
        description = prefix + arg_info['help']
        # Escape quotes in description
        description = description.replace('"', '\\"')
        param_lines.append(f'        description="{description}"')
    else:
        param_lines.append(f'        description="{prefix}{camel_name}"')

    param_lines.append("    )")

    return '\n'.join(param_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Whisper argparse arguments to CLAMS metadata parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('url', help='GitHub URL to the Whisper transcribe.py file')
    parser.add_argument('--prefix', default='(from whisper CLI) ',
                       help='Prefix to add to parameter descriptions')
    parser.add_argument('--skip', nargs='*', default=[],
                       help='Parameter names to skip (e.g., model device output_dir)')

    args = parser.parse_args()

    # Fetch file content
    print(f"Fetching content from: {args.url}", file=sys.stderr)
    content = fetch_file_content(args.url)

    # Extract argparse section
    print("Extracting argparse arguments...", file=sys.stderr)
    arguments = extract_argparse_section(content)
    print(f"Found {len(arguments)} arguments", file=sys.stderr)

    # Convert each argument
    print("\n# Generated CLAMS metadata parameters:", file=sys.stderr)
    print("# Add these to your metadata.py appmetadata() function:\n")

    skip_set = set(args.skip)
    converted_count = 0

    for arg_text in arguments:
        arg_info = parse_argument(arg_text)
        if arg_info and arg_info['name'] not in skip_set:
            clams_param = convert_to_clams_parameter(arg_info, args.prefix)
            if clams_param:
                print(clams_param)
                print()
                converted_count += 1

    print(f"\n# Converted {converted_count} parameters", file=sys.stderr)


if __name__ == '__main__':
    main()
