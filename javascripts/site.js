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

document$.subscribe(() => {
  updateProgress();
  window.removeEventListener("scroll", updateProgress);
  window.addEventListener("scroll", updateProgress, { passive: true });
});
