# icavan / systems notes

Technical notes on GPU kernels, attention systems, and distributed machine learning.

The published site is available at [icavan.github.io/random-notes](https://icavan.github.io/random-notes/).
Browse every article and supporting document in the [Archive](https://icavan.github.io/random-notes/archive/).

## Local preview

```bash
python -m pip install -r requirements-pages.txt
python scripts/prepare_site.py
mkdocs serve
```

GitHub Actions builds and deploys the site on every push to `main`.

The preparation script generates the archive from Git-tracked Markdown files,
using each document's first level-one heading as its title. New articles appear
automatically after they are added to Git; untracked local drafts are excluded.
Repository instructions, this README, site-maintenance notes, and the Home/About
pages are excluded from the article list. Nested documentation, including figure
and evidence READMEs, appears under **Supporting documents**.
