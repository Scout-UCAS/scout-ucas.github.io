"""Render the English and Chinese homepages with the Python standard library."""

import json
from hashlib import sha256
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
        "description": "Lianlei Shan’s research on multimodal reasoning and embodied foundation models, with news and selected publications.",
        "skip": "Skip to content", "nav_label": "Main navigation",
        "dark": "Switch to dark theme", "light": "Switch to light theme",
        "research_title": "Research", "news_title": "News", "publications_title": "Selected publications",
        "nav_publications": "Publications", "all_publications": "All publications on Google Scholar",
        "email_label": "Email", "portrait_alt": "WALL·E standing on rocks with both arms raised",
        "portrait_full": "Open the high-resolution WALL·E photograph",
        "code": "Code", "paper": "Paper", "preprint": "Preprint", "project": "Project page", "oral": "Oral presentation",
        "conference_abstract": "Conference abstract", "model": "Model weights",
        "figure": "View figure for", "credits": "Image credits", "top": "Back to top"
    },
    "zh": {
        "title": "Lianlei Shan | 学术主页",
        "html_lang": "zh-CN", "other_lang": "en", "other_url": "../", "other_name": "EN",
        "description": "Lianlei Shan 的学术主页，介绍多模态推理与具身智能基座模型研究，收录近期动态和代表论文。",
        "skip": "跳到正文", "nav_label": "主导航",
        "dark": "切换为深色主题", "light": "切换为浅色主题",
        "research_title": "研究内容", "news_title": "近期动态", "publications_title": "代表论文",
        "nav_publications": "论文", "all_publications": "在 Google Scholar 查看全部论文",
        "email_label": "邮件联系", "portrait_alt": "瓦力站在碎石上，举起双臂",
        "portrait_full": "查看瓦力的高清照片",
        "code": "源码", "paper": "论文原文", "preprint": "预印本", "project": "项目页", "oral": "口头报告",
        "conference_abstract": "会议摘要", "model": "模型权重",
        "figure": "查看论文配图", "credits": "图片来源", "top": "回到顶部"
    }
}


def link(url, label):
    return f'<a href="{escape(url, quote=True)}">{escape(label)}</a>'


def research_topic(topic, lang):
    topic_id = escape(topic["id"])
    heading_id = escape(topic.get("heading_id", topic["id"] + "-title"))
    return f"""
      <article id="{topic_id}" aria-labelledby="{heading_id}">
        <h3 id="{heading_id}">{escape(topic['name'][lang])}</h3>
        <p>{escape(topic['summary'][lang])}</p>
      </article>"""


def news_row(item, lang):
    links = " ".join(link(entry['url'], entry['name'][lang]) for entry in item['links'])
    date = escape(item['date'])
    return f"""
      <li><time datetime="{date}">{date.replace('-', '.')}</time>
        <div>{escape(item['text'][lang])} <span class="news-links">{links}</span></div></li>"""


def paper_row(paper, lang, base, profile_names, labels):
    authors = ", ".join(f"<strong>{escape(a)}</strong>" if a in profile_names else escape(a) for a in paper["authors"])
    resources = [link(paper["url"], labels["preprint"] if paper["venue"] == "arXiv" else labels["paper"])]
    for key in ("code", "preprint", "model", "project", "conference_abstract"):
        if key in paper:
            resources.append(link(paper[key], labels[key]))
    image_path = f"assets/papers/{paper['id']}.webp"
    venue = f"{escape(paper['venue'])} · {paper['year']}"
    if "oral_program" in paper:
        venue += " · " + link(paper["oral_program"], labels["oral"])
    venue_line = f'<p class="venue">{venue}</p>'
    if (DOCS / image_path).is_file():
        figure_label = escape(f"{labels['figure']} {paper['title']}", quote=True)
        visual = (
            f'<a class="paper-figure" href="{base}{image_path}" aria-label="{figure_label}">'
            f'<img src="{base}{image_path}" alt="" loading="lazy"></a>'
        )
        venue_line = f"\n          {venue_line}"
    else:
        visual, venue_line = venue_line, ""
    return f"""
      <article class="paper" id="{escape(paper['id'])}">
        {visual}
        <div class="paper-copy">{venue_line}
          <h3 lang="en">{link(paper['url'], paper['title'])}</h3>
          <p class="authors" lang="en">{authors}</p>
          <p class="paper-summary">{escape(paper['summary'][lang])}</p>
          <div class="resource-links">{''.join(resources)}</div>
        </div>
      </article>"""


def build():
    content = json.loads((HERE / "content.json").read_text(encoding="utf-8"))
    template = Template((HERE / "page.html").read_text(encoding="utf-8"))
    stylesheet_version = sha256((DOCS / "styles.css").read_bytes()).hexdigest()[:12]
    profile = content["profile"]
    profile_names = {profile["name"], *profile.get("publication_aliases", [])}
    for lang, labels in LABELS.items():
        base = "../" if lang == "zh" else ""
        output = DOCS / ("zh" if lang == "zh" else "") / "index.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        nav = [("research", "research_title"), ("news", "news_title"), ("publications", "nav_publications")]
        html = template.substitute(
            labels, base=base, name=escape(profile["name"]), stylesheet_version=stylesheet_version,
            canonical=SITE + ("zh/" if lang == "zh" else ""),
            focus=escape(profile["focus"][lang]),
            introduction="\n        ".join(
                f'<p class="lead">{escape(paragraph)}</p>'
                for paragraph in profile["introduction"][lang].split("\n\n")
            ),
            biography=escape(profile["biography"][lang]), email=escape(profile["email"]),
            scholar=escape(profile["scholar"] + ("&hl=zh-CN" if lang == "zh" else "&hl=en")),
            github=escape(profile["github"]), nav="".join(link("#" + section, labels[key]) for section, key in nav),
            research_items="".join(research_topic(topic, lang) for topic in content["research"]),
            news_items="".join(news_row(item, lang) for item in content["news"]),
            paper_items="".join(paper_row(p, lang, base, profile_names, labels) for p in content["papers"])
        )
        output.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    build()
