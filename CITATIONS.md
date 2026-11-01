# Source provenance

Record sources informing geometry, scientific claims, and implementation in
root `references.bib`. Use braced Better BibTeX-compatible entries with stable
keys, title, author/editor, URL, `urldate`, `review`, and `claim`. Review scopes:
metadata, abstract, excerpt, full-text, source-code. Limited review requires a
`note` identifying what was inspected. Do not invent missing metadata.

Place bracketed `@key` citations beside supported claims. Distinguish proposals
and mathematical deductions from claims established by a source.

Validate with `python -B tools/check_references.py` and
`python -B -m unittest discover -s tools -p test_references.py -v`.
The checker and in-memory tests are adapted from the workspace checker; they
validate bibliography syntax and citation resolution, not scientific truth.
