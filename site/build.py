"""Render the English and Chinese homepages with the Python standard library."""

import json
from html import escape
from pathlib import Path
from string import Template


HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
SITE = "https://scout-ucas.github.io/"

LABELS = {
    "en": {
        "title": "Lianlei Shan | Research",
        "html_lang": "en", "other_lang": "zh-CN", "other_url": "zh/", "other_name": "中文",
        "description": "Lianlei Shan’s academic homepage. Research, publications and software on latent reasoning, vision–language–action models and computer vision.",
        "skip": "Skip to content", "nav_label": "Main navigation",
        "dark": "Switch to dark theme", "light": "Switch to light theme",
        "research_title": "Research", "projects_title": "Research vision",
        "nav_projects": "Vision", "news_title": "News", "publications_title": "Selected publications",
        "nav_publications": "Publications", "all_publications": "All publications on Google Scholar",
        "email_label": "Email", "portrait_alt": "Portrait of Lianlei Shan",
        "code": "Code", "paper": "Paper", "preprint": "Preprint",
        "figure": "View figure for", "credits": "Figure credits", "top": "Back to top"
    },
    "zh": {
        "title": "Lianlei Shan | 学术主页",
        "html_lang": "zh-CN", "other_lang": "en", "other_url": "../", "other_name": "EN",
        "description": "Lianlei Shan 的学术主页。研究方向包括潜空间推理、视觉语言动作模型和计算机视觉，附代表论文、研究代码与实验记录。",
        "skip": "跳到正文", "nav_label": "主导航",
        "dark": "切换为深色主题", "light": "切换为浅色主题",
        "research_title": "研究方向", "projects_title": "研究愿景",
        "nav_projects": "愿景", "news_title": "近期动态", "publications_title": "代表论文",
        "nav_publications": "论文", "all_publications": "在 Google Scholar 查看全部论文",
        "email_label": "邮件联系", "portrait_alt": "Lianlei Shan 的头像",
        "code": "源码", "paper": "论文原文", "preprint": "预印本",
        "figure": "查看论文配图", "credits": "配图来源", "top": "回到顶部"
    }
}


def link(url, label):
    return f'<a href="{escape(url, quote=True)}">{escape(label)}</a>'


def project_card(project, lang):
    return f"""
      <article class="project" id="{escape(project['id'])}">
        <h3>{escape(project['name'][lang])}</h3>
        <p>{escape(project['summary'][lang])}</p>
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
            project_items="".join(project_card(p, lang) for p in content["projects"]),
            news_items="".join(news_row(item, lang) for item in content["news"]),
            paper_items="".join(paper_row(p, lang, base, profile["name"], labels) for p in content["papers"])
        )
        output.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    build()
