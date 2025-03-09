ifeq ($(OS_TYPE),)
    OS := $(shell uname)
    ifeq ($(OS), Linux)
		RM = rm
    else ifeq ($(OS), Darwin)
		RM = rm
    else
        RM = del
    endif
endif

report: report.tex reference.bib Makefile
	xelatex report.tex
	bibtex report.aux
	xelatex report.tex
	xelatex report.tex

clean:
	$(RM) report.aux
	$(RM) report.bbl
	$(RM) report.blg
	$(RM) report.log
	$(RM) report.out
	$(RM) report.toc