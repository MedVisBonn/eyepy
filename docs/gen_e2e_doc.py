"""Generate the E2E file documentatinon pages."""

import inspect
from pathlib import Path
from typing import List

import mkdocs_gen_files

from eyepy.io.he import e2e_format
from eyepy.io.he import e2e_reader

excluded = ['src/eyepy/config.py']

types = e2e_format.__all_types__
file_structures = e2e_format.__e2efile_structures__

import re


def clean_docstring(docstring):
    """Cleans a Python docstring for use in a Markdown file.

    :param docstring: A string containing the docstring to be cleaned.
    :return: A cleaned version of the input docstring.
    """
    # Remove leading and trailing whitespace
    docstring = docstring.strip()

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


def _get_parses_to(type_annotation):
    """Convert type annotations to markdown links."""
    if type_annotation.__name__ == 'List':
        for t in type_annotation.__args__:
            return f'List[{_get_parses_to(t)}]'
    if type_annotation in types:
        return f'[{type_annotation.__name__}](../../he_e2e_types/{type_annotation.__name__})'
    elif type_annotation in file_structures:
        return f'[{type_annotation.__name__}](../{type_annotation.__name__})'

    return str(type_annotation.__name__)


# Collect types data for the overview pages
types_data = {}
# Generate Documentation for all Types used in the E2E format
nav = mkdocs_gen_files.Nav()
for t in types:
    # Extract information from the docstring
    doc = clean_docstring(t.__doc__)
    doc_title = doc.splitlines()[0]
    t_id = int(t.__name__.lstrip('Type'))
    type_occ = [k for k, v in e2e_reader.type_occurence.items() if t_id in v]
    type_occ = [f'[{t}](formats/he_e2e_types/{t})' for t in type_occ]
    type_occ = ', '.join(type_occ)
    size = [
        l.lstrip('Size: ') for l in doc.splitlines() if l.startswith('Size: ')
    ][0]
    text = doc.split('Notes:')[1]

    # Name of the documentation file
    doc_path = Path(f'{t.__name__}.md')
    # Path to the documentation file
    full_doc_path = Path(f'formats/he_e2e_types/', doc_path)

    parts = (f'{t.__name__}', )
    nav[parts] = doc_path.as_posix()

    # Collect types data for the overview pages
    types_data[t_id] = {
        'size': size,
        'content': doc_title,
        'description': doc.splitlines()[1],
    }

    with mkdocs_gen_files.open(full_doc_path, 'w') as fd:
        print(
            f'# {t.__name__} ({doc_title})\n**Size:** {size} \n\n **Occurence:** {type_occ}\n',
            file=fd)
        print(f'\n{text}\n', file=fd)
        print('Offset|Name|Size|Parses to|Description', file=fd)
        print('---|----|----|----|-------------', file=fd)
        offset = 0
        type_annotations = t.__annotations__
        for name, f in t.__dataclass_fields__.items():
            description = f.metadata['subcon'].docs
            try:
                size = f.metadata['subcon'].sizeof()
            except:
                size = 'variable'
            t_annotation = type_annotations[name]
            parses_to = _get_parses_to(t_annotation)

            print(f'{offset}|{name}|{size}|{parses_to}|{description}', file=fd)
            offset = offset + size if (type(size) == int
                                       and type(offset) == int) else 'variable'

    #mkdocs_gen_files.set_edit_path(full_doc_path, path)  #

with mkdocs_gen_files.open('formats/he_e2e_types/SUMMARY.md', 'w') as nav_file:
    nav_file.writelines(nav.build_literate_nav())

# Generate Documentation for all E2E File Structures
nav = mkdocs_gen_files.Nav()
for t in file_structures:
    # Extract information from the docstring
    doc = clean_docstring(t.__doc__)
    doc_title = doc.splitlines()[0]
    size = [
        l.lstrip('Size: ') for l in doc.splitlines() if l.startswith('Size: ')
    ][0]
    text = doc.split('Notes:')[1]

    # Name of the documentation file
    doc_path = Path(f'{t.__name__}.md')
    # Path to the documentation file
    full_doc_path = Path(f'formats/he_e2e_structures/', doc_path)

    parts = (f'{doc_title}', )
    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, 'w') as fd:
        print('', file=fd)
        print('Offset|Name|Size|Parses to|Description', file=fd)
        print('---|----|----|----|-------------', file=fd)
        offset = 0
        type_annotations = t.__annotations__
        for name, f in t.__dataclass_fields__.items():
            description = f.metadata['subcon'].docs
            try:
                size = f.metadata['subcon'].sizeof()
            except:
                size = 'variable'
            t_annotation = type_annotations[name]
            parses_to = _get_parses_to(t_annotation)
            print(f'{offset}|{name}|{size}|{parses_to}|{description}', file=fd)
            offset = offset + size if (type(size) == int
                                       and type(offset) == int) else 'variable'

with mkdocs_gen_files.open('formats/he_e2e_structures/SUMMARY.md',
                           'w') as nav_file:
    nav_file.writelines(nav.build_literate_nav())


# Generate Documentation of E2E Hierarchy layers (E2EFile, E2EPatient, E2EStudy, E2ESeries, E2ESlice)
nav = mkdocs_gen_files.Nav()
for layer, types in e2e_reader.type_occurence.items():
    # Name of the documentation file
    doc_path = Path(f'{layer}.md')
    # Path to the documentation file
    full_doc_path = Path(f'formats/he_e2e_hierarchy/', doc_path)

    parts = (f'{layer}', )
    nav[parts] = doc_path.as_posix()

    #with mkdocs_gen_files.open(full_doc_path, "w") as fd:
    with mkdocs_gen_files.open(full_doc_path, 'w') as fd:
        print('', file=fd)
        print('|Type ID|Content|Size|Description|', file=fd)
        print('|---|---|---|--------------------|', file=fd)
        offset = 0
        for t in types:
            try:
                content = types_data[t]['content']
                size = types_data[t]['size']
                description = types_data[t]['description'].strip('\n')
            except:
                content = ''
                size = ''
                description = ''

            print(
                f'|[{t} :material-link:](/formats/he_e2e_types/Type{t})|{content}|{size}|{description}|',
                file=fd)

with mkdocs_gen_files.open('formats/he_e2e_hierarchy/SUMMARY.md',
                           'w') as nav_file:
    nav_file.writelines(nav.build_literate_nav())
