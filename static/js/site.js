document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    window.lucide.createIcons({ strokeWidth: 1.8 });
  }

  const toggle = document.querySelector(".nav-toggle");
  const links = document.querySelector(".nav-links");

  if (toggle && links) {
    toggle.addEventListener("click", () => {
      const open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    });

    links.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        links.classList.remove("open");
        toggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  const copyButton = document.querySelector("[data-copy-target]");
  if (copyButton) {
    copyButton.addEventListener("click", async () => {
      const target = document.getElementById(copyButton.dataset.copyTarget);
      if (!target) return;

      const label = copyButton.querySelector("span");
      try {
        await navigator.clipboard.writeText(target.innerText);
        if (label) label.textContent = "Copied";
        window.setTimeout(() => {
          if (label) label.textContent = "Copy";
        }, 1800);
      } catch {
        if (label) label.textContent = "Select text";
      }
    });
  }
});
