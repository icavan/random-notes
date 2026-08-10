# Repository Instructions

## GitHub Markdown math

- Use fenced `math` blocks for every display equation:

  ````markdown
  ```math
  y = H_D x
  ```
  ````

- Do not use `$$ ... $$` display delimiters in repository Markdown. GitHub parses Markdown before rendering math, so a formula line beginning with `=` can become a Setext heading and a line beginning with `+` or `-` can become a list item.
- Do not use the `\operatorname{...}` LaTeX macro. GitHub rejects it as an
  unsupported macro in this repository's renderer. Use `\mathrm{...}` for
  named functions such as `\mathrm{softmax}` and `\mathrm{logaddexp}`.
- Keep LaTeX delimiters out of Markdown headings. Write a plain-text heading and put any formal expression in the following paragraph or fenced `math` block.
- Before committing Markdown with formulas, run
  `rg -n '^\$\$$|^#{1,6} .*\$|\\operatorname' <edited-markdown-files>`.
  Any match must be removed or explicitly justified.
- After pushing, inspect the actual GitHub Preview rather than trusting a local renderer. Confirm that each display equation is represented by GitHub's display math renderer and that no raw `$` delimiters remain in ordinary rendered text.
- When adding a document with local images, verify every relative image path from the document's repository location.

## Change hygiene

- Stage only files that belong to the requested change. Preserve unrelated tracked modifications and untracked files.
