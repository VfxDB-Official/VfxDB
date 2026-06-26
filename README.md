# VfxDB — Project Website

This branch (`gh-pages`) contains **only** the project website, served by GitHub Pages at
<https://vfxdb-official.github.io/VfxDB/>.

- `index.html`, `static/` — the website source.
- `.nojekyll` — tells GitHub Pages to serve files as-is (no Jekyll processing).

The training and inference **code** lives on the `main` branch, not here.

To update the site, edit `index.html` / `static/` on this branch and push; GitHub Pages
redeploys automatically.
