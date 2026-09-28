# Design QA

## Visual source of truth

- Selected direction: option 3, revised with option 1's sans-serif typography and the real `icavan` GitHub avatar.
- Source: `/Users/van/.codex/generated_images/01a0e634-06c0-7de2-bca1-19077a2530a1/exec-8ea9f9c3-782e-4047-a6c0-f68a414281be.png`
- Source dimensions: 1487 × 1058 px.
- Implementation capture: `/Users/van/.codex/visualizations/2026/09/28/01a0e634-06c0-7de2-bca1-19077a2530a1/icavan-editorial-home-final-1487.png`
- Capture viewport: 1487 × 1058 CSS px, light theme, homepage at scroll position 0.
- Density normalization: both comparison inputs are 1487 × 1058 px; no scaling was required.

## Evidence

- Full-view comparison: `/Users/van/.codex/visualizations/2026/09/28/01a0e634-06c0-7de2-bca1-19077a2530a1/icavan-editorial-source-vs-build.png`
- Latest-note focused comparison: `/Users/van/.codex/visualizations/2026/09/28/01a0e634-06c0-7de2-bca1-19077a2530a1/icavan-editorial-latest-source-vs-build.png`
- Mobile capture: `/Users/van/.codex/visualizations/2026/09/28/01a0e634-06c0-7de2-bca1-19077a2530a1/icavan-editorial-mobile.png`

## Comparison passes

1. Initial implementation pass found three fidelity or behavior issues: the Material header contract was missing and caused console errors, search appeared as an inline Material component instead of an editorial overlay, and the featured image was too tall relative to the selected direction.
2. The implementation added a hidden compatible header hook, a custom full-screen search treatment, an explicit theme control, and a wider 1.9:1 featured-image frame.
3. Final comparison confirms the intended fixed profile rail, restrained rust accent, warm off-white surface, sans-serif hierarchy, hairline dividers, real GitHub avatar, two-column feature story, and compact recent-note index.
4. The feature illustration intentionally uses the repository's real DeepSeek architecture diagram instead of the mockup placeholder. Dates and note summaries use the repository's actual material.

## Functional and responsive checks

- Desktop homepage: no horizontal overflow; all images load; search opens and closes; light/dark theme toggle works.
- Mobile homepage: single-column feature and note layout; no horizontal overflow; navigation and utility controls remain available.
- Article page: no horizontal overflow, all inspected images load, and 108 MathJax containers render on the Fast Hadamard Transform article.
- Browser console: no errors on the final homepage or inspected article page.
- Accessibility: semantic landmark structure, labelled navigation and utility controls, descriptive image alt text, visible link states, reduced-motion handling, and practical mobile targets are present.

## Findings

- P0: none.
- P1: none.
- P2: none.

final result: passed
