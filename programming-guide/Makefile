# Croqtile Programming Guide
# Build convention copied from gemm_sp_paper/sc26-pap1104 and
# croqtile-tuner-paper/sigopsatc2026-35: pdfLaTeX -> BibTeX -> pdfLaTeX x2,
# then copy the finished PDF into output/.  No Tectonic, no latexmk.

ENGINE     := pdflatex
FLAGS      := -interaction=nonstopmode -halt-on-error
OUTPUT_DIR := output

PDF_EBOOK  := $(OUTPUT_DIR)/croqtile-programming-guide.pdf
PDF_MANUAL := $(OUTPUT_DIR)/croqtile-programming-guide-manual.pdf
PDF_MEMOIR := $(OUTPUT_DIR)/croqtile-programming-guide-memoir.pdf

SOURCES := main.tex main-manual.tex main-memoir.tex body.tex croqguide.sty \
           references.bib $(wildcard preamble/*.tex) $(wildcard content/*.tex) \
           $(wildcard build/units/*.tex) $(wildcard figures/*.png) \
           UNIT-REGISTRY.md tools/build_guide.py tools/figure_editorial.py tools/prose_editorial.py \
           editorial/figure-captions.json editorial/prose-polish.json editorial/narrative-polish.json $(wildcard editorial/getting-started/*.md)

.PHONY: all ebook manual memoir clean distclean check

# Default entry point: plain `make` builds edition A from main.tex.
# The two optional editions keep the same recipe but use suffixed entry
# points (main-manual.tex, main-memoir.tex) and suffixed PDF names.
.DEFAULT_GOAL := ebook

ebook:  $(PDF_EBOOK)

# Opt-in: build all three editions at once.
all: ebook manual memoir

manual: $(PDF_MANUAL)
memoir: $(PDF_MEMOIR)

define BUILD
	$(ENGINE) $(FLAGS) $(1).tex
	bibtex $(1)
	$(ENGINE) $(FLAGS) $(1).tex
	$(ENGINE) $(FLAGS) $(1).tex
	mkdir -p $(OUTPUT_DIR)
	cp $(1).pdf $(2)
endef

$(PDF_EBOOK): $(SOURCES)
	$(call BUILD,main,$@)

$(PDF_MANUAL): $(SOURCES)
	$(call BUILD,main-manual,$@)

$(PDF_MEMOIR): $(SOURCES)
	$(call BUILD,main-memoir,$@)

check: all
	@for p in $(PDF_EBOOK) $(PDF_MANUAL) $(PDF_MEMOIR); do \
	  echo "== $$p"; pdfinfo $$p | grep -E '^(Title|Pages|Page size|Producer):'; \
	done

clean:
	rm -f main*.aux main*.bbl main*.blg main*.log main*.out main*.toc \
	      main*.lof main*.lot main*.lol main*.pdf main*.synctex.gz
	rm -f $(OUTPUT_DIR)/*.pdf

distclean: clean
	rmdir $(OUTPUT_DIR) 2>/dev/null || true
