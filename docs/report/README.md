# Report source

The report uses the official NeurIPS 2026 LaTeX style in preprint mode, with the course-report notice replacing the conference notice. The style file is otherwise unchanged. No custom margins, fonts, colored text, shaded tables, or decorative headings are added.

Official author kit: https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip

- `report.tex`: report prose and citations.
- `*_table.tex`: tables regenerated from saved experiment JSONs.
- `neurips_2026.sty`: official style file from the author kit.
- `metrics.tex`: generated-input marker.

Run `python scripts/build_final_report.py` from the repository root. Tectonic must be available on PATH, and the Python dependencies are in `requirements-report.txt`. The builder regenerates the monochrome training figure and tables, compiles the LaTeX source, and writes `output/pdf/PA2_Final_Report.pdf`. It checks the assignment's eight-page limit.
