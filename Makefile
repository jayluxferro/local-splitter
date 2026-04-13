.PHONY: paper clean arxiv figures

PAPER_DIR := paper
TEX      := $(PAPER_DIR)/paper.tex
BBL      := $(PAPER_DIR)/paper.bbl
BIB      := $(PAPER_DIR)/bibliography.bib
FIG_DIR  := $(PAPER_DIR)/figures
PDF      := $(PAPER_DIR)/paper.pdf
ARXIV_ZIP := local-splitter-arxiv.zip

# Build the PDF (pdflatex + bibtex + 2x pdflatex)
paper: figures
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode paper.tex
	cd $(PAPER_DIR) && bibtex paper
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode paper.tex
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode paper.tex
	@echo "Built $(PDF) ($$(wc -c < $(PDF)) bytes)"

# Generate figures from eval data
figures:
	uv run python scripts/gen_figures.py

# Package for arXiv upload
# arXiv needs: .tex, .bbl, figures — no .bib, no .pdf, no aux/log
arxiv: paper
	rm -f $(ARXIV_ZIP)
	zip -j $(ARXIV_ZIP) $(TEX) $(BBL)
	cd $(PAPER_DIR) && zip -g ../$(ARXIV_ZIP) figures/*.pdf
	@echo "Created $(ARXIV_ZIP):"
	@zipinfo -1 $(ARXIV_ZIP) | sed 's/^/  /'
	@echo "Upload this file to https://arxiv.org/submit"

# Remove build artifacts
clean:
	cd $(PAPER_DIR) && rm -f *.aux *.log *.out *.toc *.blg *.synctex.gz *.fls *.fdb_latexmk
	rm -f $(ARXIV_ZIP)
