"""Render the English and Chinese homepages with the Python standard library."""

import json
from html import escape
from pathlib import Path
from string import Template
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
SITE = "https://scout-ucas.github.io/"
REPO = "https://github.com/Scout-UCAS/scout-ucas.github.io"

LABELS = {
    "en": {
        "title": "Lianlei Shan | Research",
        "html_lang": "en", "other_lang": "zh-CN", "other_url": "zh/", "other_name": "中文",
        "description": "Lianlei Shan’s academic homepage. Research, publications and software on latent reasoning, vision–language–action models and computer vision.",
        "skip": "Skip to content", "nav_label": "Main navigation",
        "dark": "Switch to dark theme", "light": "Switch to light theme",
        "research_title": "Research", "projects_title": "Research software",
        "nav_projects": "Projects", "news_title": "News", "publications_title": "Selected publications",
        "nav_publications": "Publications", "all_publications": "All publications on Google Scholar",
        "email_label": "Email", "portrait_alt": "Portrait of Lianlei Shan",
        "code": "Code", "report": "Experiment report", "paper": "Paper", "preprint": "Preprint",
        "executor": "Arithmetic executor", "readouts": "Expected readouts", "swap": "swap",
        "blue": "Blue pencil", "red": "Red pencil", "simulator": "CPU simulator · matched seed",
        "figure": "View figure for", "credits": "Figure credits", "top": "Back to top"
    },
    "zh": {
        "title": "Lianlei Shan | 学术主页",
        "html_lang": "zh-CN", "other_lang": "en", "other_url": "../", "other_name": "EN",
        "description": "Lianlei Shan 的学术主页。研究方向包括潜空间推理、视觉语言动作模型和计算机视觉，附代表论文、研究代码与实验记录。",
        "skip": "跳到正文", "nav_label": "主导航",
        "dark": "切换为深色主题", "light": "切换为浅色主题",
        "research_title": "研究方向", "projects_title": "研究项目",
        "nav_projects": "项目", "news_title": "近期动态", "publications_title": "代表论文",
        "nav_publications": "论文", "all_publications": "在 Google Scholar 查看全部论文",
        "email_label": "邮件联系", "portrait_alt": "Lianlei Shan 的头像",
        "code": "源码", "report": "实验报告", "paper": "论文原文", "preprint": "预印本",
        "executor": "算术执行器", "readouts": "预期读出", "swap": "交换",
        "blue": "蓝色铅笔", "red": "红色铅笔", "simulator": "CPU 模拟器 · 相同评测种子",
        "figure": "查看论文配图", "credits": "配图来源", "top": "回到顶部"
    }
}


def link(url, label):
    return f'<a href="{escape(url, quote=True)}">{escape(label)}</a>'


def project_figure(project_id, base, labels):
    if project_id == "ct-rl":
        return f"""
        <figure class="project-figure arithmetic">
          <figcaption>{labels['executor']}</figcaption>
          <div class="registers"><code>(2, 3)</code><span class="operator">{labels['swap']}</span><code>(3, 2)</code></div>
          <div class="readouts"><span>{labels['readouts']}</span><code>left = 3</code><code>diff = 1</code></div>
        </figure>"""
    return f"""
        <figure class="project-figure image-pair">
          <figcaption>{labels['simulator']}</figcaption>
          <div class="samples">
            <div><img src="{base}assets/responseguard-blue.png" alt="" width="64" height="64"><span>{labels['blue']}</span></div>
            <div><img src="{base}assets/responseguard-red.png" alt="" width="64" height="64"><span>{labels['red']}</span></div>
          </div>
        </figure>"""


def project_card(project, lang, base, labels):
    source = f"{REPO}/tree/main/{quote(project['source'])}"
    report = f"{REPO}/blob/main/{quote(project['report'])}"
    return f"""
      <article class="project" id="{escape(project['id'])}">
        {project_figure(project['id'], base, labels)}
        <div class="project-copy">
          <h3>{escape(project['name'])}</h3>
          <p class="subtitle">{escape(project['subtitle'][lang])}</p>
          <p>{escape(project['summary'][lang])}</p>
          <p class="validation">{escape(project['validation'][lang])}</p>
          <div class="resource-links">{link(source, labels['code'])}{link(report, labels['report'])}</div>
        </div>
      </article>"""


def news_row(item, lang):
    links = " ".join(link(entry['url'], entry['name'][lang]) for entry in item['links'])
    date = escape(item['date'])
    return f"""
      <li><time datetime="{date}">{date.replace('-', '.')}</time>
        <div>{escape(item['text'][lang])} <span class="news-links">{links}</span></div></li>"""


def paper_row(paper, lang, base, name, labels):
    authors = ", ".join(f"<strong>{escape(a)}</strong>" if a == name else escape(a) for a in paper["authors"])
    resources = [link(paper["url"], labels["preprint"] if paper["venue"] == "arXiv" else labels["paper"])]
    for key in ("code", "preprint"):
        if key in paper:
            resources.append(link(paper[key], labels[key]))
    image_path = f"assets/papers/{paper['id']}.webp"
    venue = f"{escape(paper['venue'])} · {paper['year']}"
    venue_line = f'<p class="venue">{venue}</p>'
    if (DOCS / image_path).is_file():
        figure_label = escape(f"{labels['figure']} {paper['title']}", quote=True)
        visual = (
            f'<a class="paper-figure" href="{base}{image_path}" aria-label="{figure_label}">'
            f'<img src="{base}{image_path}" alt="" loading="lazy"></a>'
        )
    else:
        visual, venue_line = venue_line, ""
    return f"""
      <article class="paper" id="{escape(paper['id'])}">
        {visual}
        <div class="paper-copy">
          {venue_line}
          <h3 lang="en">{link(paper['url'], paper['title'])}</h3>
          <p class="authors" lang="en">{authors}</p>
          <p class="paper-summary">{escape(paper['summary'][lang])}</p>
          <div class="resource-links">{''.join(resources)}</div>
        </div>
      </article>"""


def build():
    content = json.loads((HERE / "content.json").read_text(encoding="utf-8"))
    template = Template((HERE / "page.html").read_text(encoding="utf-8"))
    profile = content["profile"]
    for lang, labels in LABELS.items():
        base = "../" if lang == "zh" else ""
        output = DOCS / ("zh" if lang == "zh" else "") / "index.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        nav = [("research", "research_title"), ("projects", "nav_projects"),
               ("news", "news_title"), ("publications", "nav_publications")]
        research = (
            f"<article><h3>{escape(item['name'][lang])}</h3><p>{escape(item['summary'][lang])}</p></article>"
            for item in content["research"]
        )
        html = template.substitute(
            labels, base=base, name=escape(profile["name"]),
            canonical=SITE + ("zh/" if lang == "zh" else ""),
            focus=escape(profile["focus"][lang]), introduction=escape(profile["introduction"][lang]),
            biography=escape(profile["biography"][lang]), email=escape(profile["email"]),
            scholar=escape(profile["scholar"] + ("&hl=zh-CN" if lang == "zh" else "&hl=en")),
            github=escape(profile["github"]), nav="".join(link("#" + section, labels[key]) for section, key in nav),
            research_items="\n".join(research),
            project_items="".join(project_card(p, lang, base, labels) for p in content["projects"]),
            news_items="".join(news_row(item, lang) for item in content["news"]),
            paper_items="".join(paper_row(p, lang, base, profile["name"], labels) for p in content["papers"])
        )
        output.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    build()
