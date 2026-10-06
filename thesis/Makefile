LATEXMK = latexmk
LATEX = xelatex
OUT_DIR = output
MAIN = main
CLEAN_EXTENSIONS = aux log out toc bbl blg synctex.gz run.xml bcf pdf sta

LATEXMK_OPTS = -pdfxe -output-directory=$(OUT_DIR)
LATEXMK_NO_BIBER_OPTS = -pdfxe -output-directory=$(OUT_DIR) -bibtex-

.PHONY: all quick clean final presentation thesis

$(OUT_DIR):
	if not exist $(OUT_DIR) mkdir $(OUT_DIR)

build: presentation $(OUT_DIR)
	$(LATEXMK) $(LATEXMK_OPTS) $(MAIN).tex

thesis: $(OUT_DIR)
	$(LATEX) -output-directory=$(OUT_DIR) $(MAIN).tex
	$(LATEX) -output-directory=$(OUT_DIR) $(MAIN).tex

quick: $(OUT_DIR)
	$(LATEX) -output-directory=$(OUT_DIR) $(MAIN).tex

clean:
	if exist $(OUT_DIR) rmdir /s /q $(OUT_DIR)

final: clean build

presentation:
	if not exist Presentation\output mkdir Presentation\output
	$(LATEXMK) -pdfxe -cd -output-directory=output Presentation/beamer.tex