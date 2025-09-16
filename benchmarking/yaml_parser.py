import yaml
# from jsonschema import validate

"""Utilities for validating and gently auto-fixing YAML snippets.

This module exposes:
- SCHEMA: a minimal JSON-schema-like structure describing the expected YAML shape.
- validate_and_fix_yaml: loads YAML text, and on common parse errors, attempts small
  string-level fixes (trimming artifacts, quoting scalar values, splitting joined keys)
  before retrying the parse.
"""

SCHEMA = {
    'type': 'object',
    'properties': {
        'area': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'type': {'type': 'string'},
            },
            'required': ['name', 'type']
        }
    },
    'required': ['area']
}


def validate_and_fix_yaml(yaml_text):
    """Parse YAML text and attempt simple auto-fixes on common parse errors.

    This function tries to be resilient to frequent generation/formatting artifacts:
    - Removes trailing model tokens like ``</s>`` and ``<|endoftext|>``.
    - On ``yaml.parser.ParserError``: trims whitespace on lines that look like
      collection headers (e.g., "entities", "relations") and retries.
    - On ``yaml.composer.ComposerError``: auto-quotes scalar values after a key
      named ``value:`` and retries.
    - On ``yaml.scanner.ScannerError``: inserts a newline before an inline ``id:``
      that appears on the same line as another key and retries.

    Args:
        yaml_text (str): Raw YAML text to be parsed.

    Returns:
        Any | None: The Python object produced by ``yaml.safe_load`` (typically a
        ``dict`` for this schema), or ``None`` if an error was handled but no fix
        condition applied (and thus no recursive retry returned a value).

    Notes:
        - Schema validation with ``jsonschema.validate`` is present but commented
          out; enable it if you want strict conformance to ``SCHEMA`` after load.
        - Errors are printed to stdout when auto-fixes are attempted.

    Raises:
        Any exception from PyYAML not explicitly handled here will propagate.

    """
    yaml_text = yaml_text.replace('</s>', '')
    yaml_text = yaml_text.replace('<|endoftext|>','')
    try:
        result = yaml.safe_load(yaml_text)
        # validate(instance=result, schema=SCHEMA)
        return result
    except yaml.parser.ParserError as e:
        print(f"fixing error: {e}")
        line_num = e.problem_mark.line
        # column_num = e.problem_mark.column
        lines = yaml_text.split('\n')

        misformatted_line = lines[line_num]
        if "entities" or "relations" in lines[line_num]:
            corrected_line = misformatted_line.strip()
            yaml_text = yaml_text.replace(misformatted_line, corrected_line)
            return validate_and_fix_yaml(yaml_text)
    except yaml.composer.ComposerError as e:
        print(f"fixing error: {e}")
        line_num = e.problem_mark.line
        # column_num = e.problem_mark.column
        lines = yaml_text.split('\n')

        if "value" in lines[line_num]:
            tag = lines[line_num].split(":")
            tag_value = tag[1].strip()
            fixed_tag_value = "\"" + tag_value + "\""
            yaml_text = yaml_text.replace(tag_value, fixed_tag_value)
            return validate_and_fix_yaml(yaml_text)

    except yaml.scanner.ScannerError as e:
        print(f"fixing error: {e}")
        line_num = e.problem_mark.line

        # column_num = e.problem_mark.column
        lines = yaml_text.split('\n')

        misformatted_line = lines[line_num]
        if "value" and "id" in lines[line_num]:
            corrected_line = misformatted_line.replace("id:", "\n id:")
            yaml_text = yaml_text.replace(misformatted_line, corrected_line)
            return validate_and_fix_yaml(yaml_text)


