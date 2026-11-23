# Lianlei Shan

[Academic homepage](https://scout-ucas.github.io/) · [中文主页](https://scout-ucas.github.io/zh/) · [Google Scholar](https://scholar.google.com/citations?user=V5YWfWgAAAAJ)

## Research software

- **CT-RL** trains a reusable latent state by testing several continuations of the same prefix. [Source and reproduction instructions](projects/ct-rl/) · [CPU smoke report](projects/ct-rl/reports/smoke/REPORT.md)
- **ResponseGuard** evaluates prompt responsiveness and conditional diversity in image generation with paired prompts. [Source and reproduction instructions](projects/responseguard/) · [CPU smoke report](projects/responseguard/docs/Smoke运行报告.md)

Each project has its own Python environment, training configurations, tests and experiment records.

## Editing the homepage

The profile, projects, news and selected papers share a bilingual content file. Each paper links to its proceedings entry or preprint. [Content](site/content.json) · [Layout](site/page.html) · [Styles](docs/styles.css)

After editing the content or template, run the following commands from the repository root to rebuild and preview both pages.

```sh
python3 site/build.py
python3 -m http.server 8000 --directory docs
```

GitHub Pages serves the committed `docs/` directory on `main`.

## Sources and credits

- The display name, research interests and photograph come from the supplied [Google Scholar profile](https://scholar.google.com/citations?user=V5YWfWgAAAAJ).
- The biographical details appear in an author's biography in *Physical Communication*. The professional contact address also appears in the ACL paper. [Author biography](https://www.sciencedirect.com/science/article/pii/S1874490726002429) · [ACL paper](https://aclanthology.org/2026.acl-long.1496.pdf)
- The CT-RL arithmetic example follows the project executor. The two ResponseGuard images come from the first matched evaluation seed for the blue/red pencil pair in the CPU simulation, with training seed 42. [Executor](projects/ct-rl/ct_rl/tasks.py) · [ResponseGuard report](projects/responseguard/docs/Smoke运行报告.md)
- The site layout was inspired by [Yingdong Hu’s academic homepage](https://yingdong-hu.github.io/).
