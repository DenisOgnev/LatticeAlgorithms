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
) 

file(COPY ${PROJECT_BINARY_DIR}/main.pdf DESTINATION ${CMAKE_SOURCE_DIR}/docs)