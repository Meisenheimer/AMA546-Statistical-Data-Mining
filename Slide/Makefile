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

slide: slide.tex reference.bib Makefile
	xelatex slide.tex
	bibtex slide.aux
	xelatex slide.tex
	xelatex slide.tex

clean:
	$(RM) slide-blx.bib
	$(RM) slide.aux
	$(RM) slide.log
	$(RM) slide.nav
	$(RM) slide.out
	$(RM) slide.run.xml
	$(RM) slide.snm
	$(RM) slide.toc
	$(RM) slide.bbl
	$(RM) slide.blg