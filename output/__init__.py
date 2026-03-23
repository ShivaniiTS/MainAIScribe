"""
Output writers — render pipeline results as Markdown files.

All generated notes, transcripts, quality reports, and comparisons are
Markdown (.md) files with proper headers, sections, and tables.
"""

from output.markdown_writer import write_clinical_note, write_transcript
from output.comparison_writer import write_comparison

__all__ = ["write_clinical_note", "write_transcript", "write_comparison"]
