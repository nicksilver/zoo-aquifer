(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("report" "a4paper" "12pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english")))
   (TeX-run-style-hooks
    "latex2e"
    "report"
    "rep12"
    "babel"
    "packages/sleek"
    "packages/sleek-title"
    "packages/sleek-theorems"
    "packages/sleek-listings"
    "float"
    "longtable")
   (TeX-add-symbols
    "tightlist"
    "tbs")
   (LaTeX-add-labels
    "tbl-miller"
    "T_312d3"
    "fig-water-use"
    "fig-extract-map")
   (LaTeX-add-bibliographies
    "/home/nick/Zotero/better-bibtex/my-library"))
 :latex)

