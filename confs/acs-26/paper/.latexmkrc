# LaTeX-mk configuration for the paper/ folder.
#
# Routes every intermediate file (.aux, .log, .bbl, .blg, .out, .toc, ...)
# into ./temporary/ so the paper/ directory stays clean.
# The final PDF is still written to paper/ so it's easy to open.
#
# Adds ./template (cogsys.sty + cogsysapa.bst + format files) to the
# tex / bibtex / bst search paths so the moved style files still
# resolve.

$pdf_mode = 1;

# All intermediate / auxiliary files go here.
$aux_dir = './temporary';

# Final PDF stays alongside main.tex.
$out_dir = '.';

# Use pdflatex for compilation.
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# Make sure the temporary directory exists before compilation starts.
system 'mkdir -p ./temporary';

# Let pdflatex find .sty / .cls in template/ and the paper root.
ensure_path('TEXINPUTS',  './template:./:');
# Let bibtex find .bib in the paper root and template/.
ensure_path('BIBINPUTS',  './template:./:');
# Let bibtex find .bst in template/.
ensure_path('BSTINPUTS',  './template:./:');

# What `latexmk -c` (clean intermediate) should remove.
$clean_ext = 'synctex.gz acn acr alg aux bbl bcf blg fdb_latexmk fls glg glo gls idx ilg ind ist lof log lol lot nav out run.xml snm thm toc xdy';

# Bibtex configuration — natbib + .bib file in paper/.
$bibtex_use = 2;
