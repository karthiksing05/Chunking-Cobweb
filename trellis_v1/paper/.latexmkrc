# LaTeX-mk configuration for the paper/ folder.
#
# Build everything in place (paper/) so natbib can always find the .bbl
# in the same directory as main.tex. (We previously routed aux files into
# ./temporary/ to keep paper/ clean, but that broke citation resolution
# whenever pdflatex looked for main.bbl alongside main.tex.)

$pdf_mode = 1;

# Use pdflatex for compilation.
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# Force enough pdflatex passes to settle natbib cross-references when the
# aux changes (e.g. abstract edits) -- 5 is the default but be explicit so
# nobody gets stale "(?)" citations from a single-pass rebuild.
$max_repeat = 6;

# cogsys.sty + cogsysapa.bst live alongside main.tex so any LaTeX
# invocation (bare pdflatex, latexmk, LaTeX Workshop) finds them.
# format.tex / format.bib / cascade.png live in template/ as reference
# only and are not required at compile time.

# What `latexmk -c` (clean intermediate) should remove.
$clean_ext = 'synctex.gz acn acr alg aux bbl bcf blg fdb_latexmk fls glg glo gls idx ilg ind ist lof log lol lot nav out run.xml snm thm toc xdy';

# Bibtex configuration — natbib + .bib file in paper/.
$bibtex_use = 2;
