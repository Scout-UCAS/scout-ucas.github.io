(() => {
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const key = "academic-homepage-theme";
  let preference;
  try {
    preference = localStorage.getItem(key);
  } catch {
    // System appearance also works without local storage.
  }

  function applyTheme() {
    const dark = preference === "dark" || (preference !== "light" && media.matches);
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    const button = document.querySelector(".theme");
    if (button) {
      button.setAttribute("aria-label", dark ? button.dataset.light : button.dataset.dark);
      button.title = button.getAttribute("aria-label");
    }
  }

  applyTheme();
  media.addEventListener("change", applyTheme);
  document.addEventListener("DOMContentLoaded", () => {
    const button = document.querySelector(".theme");
    button.hidden = false;
    applyTheme();
    button.addEventListener("click", () => {
      preference = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      try {
        localStorage.setItem(key, preference);
      } catch {
        // Keep the selection for this page.
      }
      applyTheme();
    });
  });
})();
