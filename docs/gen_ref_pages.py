"""Generate the code reference pages.

Code reference should be grouped by package


"""

from pathlib import Path

import mkdocs_gen_files

excluded = ["src/eyepy/config.py"]

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("src/eyepy").rglob("*.py")):  #
    if str(path) in excluded:
        continue
    #print(path)
    module_path = path.relative_to(".").with_suffix("")  #
    doc_path = path.relative_to(".").with_suffix(".md")  #
    full_doc_path = Path("reference", doc_path)  #

    parts = tuple(module_path.parts)[1:]

    if parts[-1] == "__init__":  #
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:  #
        identifier = ".".join(parts)  #
        print("::: " + identifier, file=fd)  #

    mkdocs_gen_files.set_edit_path(full_doc_path, path)  #

#print([x for x in nav.build_literate_nav()])
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
