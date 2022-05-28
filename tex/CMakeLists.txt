project(Diploma)

set(PDFLATEX_COMPILER xelatex)
include(${CMAKE_SOURCE_DIR}/cmake/UseLATEX.cmake)

ADD_LATEX_DOCUMENT(main.tex 
    INPUTS 
        in/preambule.tex
        in/0.title.tex
        in/1.annotation.tex
        in/2.conventions_list.tex
        in/3.intro.tex
        in/4.main_part.tex
        in/5.conclusion.tex
        in/6.bibliography.tex
        in/7.additional.tex
        in/main_part/4.0.basic_concepts.tex
        in/main_part/4.1.formulation_of_problem.tex
        in/main_part/4.2.overview_of_sources.tex
        in/main_part/4.3.overview_of_used_instruments.tex
        in/main_part/4.4.HNF.tex
        in/main_part/4.5.CVP.tex
        in/main_part/4.6.overview_of_implementation.tex
) 
