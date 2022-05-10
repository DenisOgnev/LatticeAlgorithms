project(Diploma)

set(PDFLATEX_COMPILER xelatex)
include(${CMAKE_SOURCE_DIR}/cmake/UseLATEX.cmake)

ADD_LATEX_DOCUMENT(main.tex 
    INPUTS 
        in/preambule.tex
        in/title.tex
        in/annotation.tex
        in/conventions_list.tex
        in/1.intro.tex
        in/2.main_part.tex
        in/3.conclusion.tex
        in/bibliography.tex
        in/additional.tex
) 

file(COPY ${PROJECT_BINARY_DIR}/main.pdf DESTINATION ${CMAKE_SOURCE_DIR}/docs)