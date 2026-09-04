from pathlib import Path
import re

import mkdocs_gen_files

from eyepy.io.he import e2e_format
from eyepy.io.he import e2e_reader

types = e2e_format.__all_types__
file_structures = e2e_format.__e2efile_structures__


# The next two functions are duplicated since they are needed in gen_e2e_doc.py as well
def clean_docstring(docstring):
    """Cleans a Python docstring for use in a Markdown file.

    :param docstring: A string containing the docstring to be cleaned.
    :return: A cleaned version of the input docstring.
    """
    # Remove leading and trailing whitespace
    docstring = (docstring or '').strip()

    # Remove any leading comment characters (#) or whitespace from each line
    lines = [
        re.sub(r'^\s*#', '', line).strip() for line in docstring.split('\n')
    ]

    # Remove empty lines from the beginning and end of the docstring
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    # Remove any remaining leading or trailing whitespace from each line
    lines = [line.strip() for line in lines]

    # Join the lines back together and return the result
    return '\n'.join(lines)


def _doc_parts(obj):
    """Return a stable title, size, and description for an E2E type."""
    doc = clean_docstring(obj.__doc__)
    lines = doc.splitlines()
    title = lines[0] if lines else obj.__name__
    size = next(
        (line.removeprefix('Size:').strip() for line in lines
         if line.startswith('Size:')),
        'unknown',
    )
    description = next(
        (line for line in lines[1:]
         if line and not line.startswith(('Size:', 'Notes:'))),
        '',
    )
    return title, size, description


def _get_parses_to(type_annotation):
    """Convert type annotations to markdown links."""
    annotation_name = getattr(type_annotation, '__name__', str(type_annotation))
    if annotation_name == 'List':
        for t in type_annotation.__args__:
            return f'List[{_get_parses_to(t)}]'
    if type_annotation in types:
        return f'[{type_annotation.__name__}](../he_e2e_types/{type_annotation.__name__}.md)'
    elif type_annotation in file_structures:
        return f'[{type_annotation.__name__}]({type_annotation.__name__}.md)'

    return annotation_name

def define_env(env):
    'Hook function'

    @env.macro
    def get_structure_doc(structure_name):
        """Get the documentation for a structure."""
        names = [t.__name__ for t in file_structures]
        if structure_name in names:
            structure = file_structures[names.index(structure_name)]

            text = '|Offset|Name|Size|Parses to|Description|\n'
            text += '|---|----|----|----|-------------|\n'

            offset = 0
            type_annotations = structure.__annotations__
            for name, f in structure.__dataclass_fields__.items():
                description = f.metadata['subcon'].docs
                try:
                    size = f.metadata['subcon'].sizeof()
                except Exception:
                    size = 'variable'
                t_annotation = type_annotations[name]
                parses_to = _get_parses_to(t_annotation)
                text += f'{offset}|{name}|{size}|{parses_to}|{description}\n'
                offset = offset + size if (
                    type(size) == int and type(offset) == int) else 'variable'

            return text
        raise ValueError(
            f'Structure {structure_name} not found in structure names')

    def get_types_data():
        types_data = {}
        for t in types:
            # Extract information from the docstring
            doc_title, size, description = _doc_parts(t)
            t_id = int(t.__name__.lstrip('Type'))
            # Collect types data for the overview pages
            types_data[t_id] = {
                'size': size,
                'content': doc_title,
                'description': description,
            }
        return types_data

    @env.macro
    def get_hierarchy_doc(level_name):
        """Get the documentation for a Level in the E2E hierarchy."""
        if level_name in e2e_reader.type_occurence:
            types = e2e_reader.type_occurence[level_name]
            text = '|Type ID|Content|Size|Description|\n'
            text += '|---|---|---|--------------------|\n'
            types_data = get_types_data()
            for t in types:
                try:
                    content = types_data[t]['content']
                    size = types_data[t]['size']
                    description = types_data[t]['description'].strip('\n')
                except (KeyError, AttributeError):
                    content = ''
                    size = ''
                    description = ''
                if not content:
                    text += f'|{t}|{content}|{size}|{description}|\n'
                else:
                    text += f'|[{t} :material-link:](../he_e2e_types/Type{t}.md)|{content}|{size}|{description}|\n'

            return text
        raise ValueError(f'Level {level_name} not found in E2E hierarchy.')
