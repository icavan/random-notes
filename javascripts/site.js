const ensureProgressBar = () => {
  let progress = document.querySelector(".reading-progress");
  if (!progress) {
    progress = document.createElement("div");
    progress.className = "reading-progress";
    progress.setAttribute("aria-hidden", "true");
    document.body.appendChild(progress);
  }
  return progress;
};

const updateProgress = () => {
  const progress = ensureProgressBar();
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  const value = scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0;
  progress.style.width = `${Math.min(100, Math.max(0, value))}%`;
};

const applyTheme = (scheme) => {
  const normalized = scheme === "slate" ? "slate" : "default";
  document.body.setAttribute("data-md-color-scheme", normalized);
  document.body.setAttribute("data-md-color-primary", "custom");
  document.body.setAttribute("data-md-color-accent", "custom");
  localStorage.setItem("icavan-theme", normalized);
};

const setupThemeToggle = () => {
  const saved = localStorage.getItem("icavan-theme");
  if (saved) {
    applyTheme(saved);
  }

  const toggle = document.querySelector("[data-theme-toggle]");
  if (!toggle || toggle.dataset.ready) return;

  toggle.dataset.ready = "true";
  toggle.addEventListener("click", () => {
    const current = document.body.getAttribute("data-md-color-scheme");
    applyTheme(current === "slate" ? "default" : "slate");
  });
};

document$.subscribe(() => {
  setupThemeToggle();
  updateProgress();
  window.removeEventListener("scroll", updateProgress);
  window.addEventListener("scroll", updateProgress, { passive: true });
});
