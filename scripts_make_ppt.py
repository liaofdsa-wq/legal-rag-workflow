from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "法規資料專題_給業師簡報.pptx"


W = 13_333_333
H = 7_500_000


def emu(x: float) -> int:
    return int(x * 914400)


def text_runs(text: str, size: int = 22, bold: bool = False, color: str = "222222") -> str:
    b = "<a:b/>" if bold else ""
    return (
        f'<a:r><a:rPr lang="zh-TW" sz="{size * 100}" dirty="0">'
        f"{b}<a:solidFill><a:srgbClr val=\"{color}\"/></a:solidFill>"
        '<a:latin typeface="Microsoft JhengHei"/><a:ea typeface="Microsoft JhengHei"/>'
        f'</a:rPr><a:t>{escape(text)}</a:t></a:r>'
    )


def shape_text(
    shape_id: int,
    name: str,
    x: int,
    y: int,
    cx: int,
    cy: int,
    paragraphs: list[dict],
    fill: str | None = None,
    line: str | None = None,
    radius: bool = False,
) -> str:
    prst = "roundRect" if radius else "rect"
    fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
    line_xml = f'<a:ln><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>' if line else "<a:ln><a:noFill/></a:ln>"
    paras = []
    for p in paragraphs:
        text = p.get("text", "")
        size = p.get("size", 22)
        bold = p.get("bold", False)
        color = p.get("color", "222222")
        bullet = p.get("bullet", False)
        indent = p.get("indent", 0)
        algn = p.get("align", "l")
        mar_l = 285750 if bullet else indent
        indent_val = -228600 if bullet else 0
        bu = "<a:buChar char=\"•\"/>" if bullet else "<a:buNone/>"
        paras.append(
            f'<a:p><a:pPr algn="{algn}" marL="{mar_l}" indent="{indent_val}">{bu}</a:pPr>'
            f"{text_runs(text, size, bold, color)}<a:endParaRPr lang=\"zh-TW\" sz=\"{size * 100}\"/></a:p>"
        )
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="{escape(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
    <a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>
    {fill_xml}{line_xml}
  </p:spPr>
  <p:txBody>
    <a:bodyPr wrap="square" lIns="171450" tIns="91440" rIns="171450" bIns="91440"/>
    <a:lstStyle/>
    {''.join(paras)}
  </p:txBody>
</p:sp>
"""


def title_box(title: str, subtitle: str | None = None) -> str:
    paras = [{"text": title, "size": 30, "bold": True, "color": "153E5C"}]
    if subtitle:
        paras.append({"text": subtitle, "size": 14, "color": "5B6870"})
    return shape_text(2, "Title", emu(0.55), emu(0.28), emu(12.15), emu(0.72), paras)


def footer(n: int) -> str:
    return shape_text(
        90,
        "Footer",
        emu(0.55),
        emu(7.02),
        emu(12.15),
        emu(0.22),
        [{"text": f"金融法規資料結構化與 RAG 問答專題｜{n}", "size": 8, "color": "7A868C", "align": "r"}],
    )


def rect(shape_id: int, name: str, x: float, y: float, w: float, h: float, color: str) -> str:
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="{name}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
    <a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:ln><a:noFill/></a:ln>
  </p:spPr>
</p:sp>
"""


def slide_xml(shapes: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    {shapes}
  </p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def bullet_slide(num: int, title: str, bullets: list[str], subtitle: str | None = None) -> str:
    paragraphs = [{"text": b, "size": 20, "bullet": True, "color": "263238"} for b in bullets]
    return slide_xml(
        rect(70, "Accent", 0, 0, 13.33, 0.13, "2C7A7B")
        + title_box(title, subtitle)
        + shape_text(3, "Bullets", emu(0.9), emu(1.35), emu(11.5), emu(4.9), paragraphs)
        + footer(num)
    )


def load_stats() -> dict:
    summary = ROOT / "08_leaf_json_embedding_問答" / "法規資料_md_clean_leaf_json" / "data" / "summary" / "file_summary.json"
    with summary.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {
        "files": len(rows),
        "nodes": sum(r.get("node_count", 0) for r in rows),
        "leaves": sum(r.get("leaf_count", 0) for r in rows),
        "cells": sum(r.get("table_cell_count", 0) for r in rows),
        "chunks": sum(r.get("table_chunk_count", 0) for r in rows),
    }


def build_slides() -> list[str]:
    s = load_stats()
    slides = []
    slides.append(
        slide_xml(
            rect(70, "Top", 0, 0, 13.33, 0.18, "2C7A7B")
            + shape_text(
                2,
                "Cover",
                emu(0.85),
                emu(1.35),
                emu(11.6),
                emu(2.05),
                [
                    {"text": "金融法規資料結構化與 RAG 問答專題", "size": 34, "bold": True, "color": "153E5C"},
                    {"text": "PDF 法規資料整理、結構化切段、Embedding 與問答展示", "size": 18, "color": "455A64"},
                    {"text": "給業師簡報", "size": 15, "color": "2C7A7B"},
                ],
            )
            + shape_text(
                3,
                "CoverNotes",
                emu(0.9),
                emu(4.65),
                emu(11.5),
                emu(1.1),
                [
                    {"text": "資料來源：法規資料_整理保留_20260412", "size": 15, "color": "5B6870"},
                    {"text": "說明重點：資料處理流程、RAG 檢索設計、目前成果與後續方向", "size": 15, "color": "5B6870"},
                ],
                fill="F4F7F8",
                line="D8E1E5",
                radius=True,
            )
            + footer(1)
        )
    )
    slides.append(
        bullet_slide(
            2,
            "專案目標",
            [
                "將金融與資安相關法規 PDF 轉為可檢索、可追溯的結構化資料。",
                "保留章、節、條、款、項等法規階層，降低固定字數切段造成的語意斷裂。",
                "建立多種 embedding 模式，支援比較不同檢索策略的效果。",
                "提供本機問答 App 與批次問答流程，方便展示 RAG 檢索與回答生成。",
            ],
        )
    )
    slides.append(
        slide_xml(
            rect(70, "Accent", 0, 0, 13.33, 0.13, "2C7A7B")
            + title_box("資料範圍", "金融資訊安全、電子金融、作業韌性、供應鏈風險、新興科技與內控規範")
            + shape_text(3, "Stats1", emu(0.9), emu(1.35), emu(2.6), emu(1.2), [{"text": "63", "size": 32, "bold": True, "color": "153E5C"}, {"text": "原始 PDF 檔", "size": 13, "color": "455A64"}], fill="F4F7F8", line="D8E1E5", radius=True)
            + shape_text(4, "Stats2", emu(3.85), emu(1.35), emu(2.6), emu(1.2), [{"text": "46", "size": 32, "bold": True, "color": "153E5C"}, {"text": "精修 Markdown", "size": 13, "color": "455A64"}], fill="F4F7F8", line="D8E1E5", radius=True)
            + shape_text(5, "Stats3", emu(6.8), emu(1.35), emu(2.6), emu(1.2), [{"text": str(s["files"]), "size": 32, "bold": True, "color": "153E5C"}, {"text": "已彙整檔案摘要", "size": 13, "color": "455A64"}], fill="F4F7F8", line="D8E1E5", radius=True)
            + shape_text(6, "Scope", emu(0.9), emu(3.0), emu(11.5), emu(2.6), [
                {"text": "資料主題包含證券、期貨、保險與金融機構相關規範，特別聚焦資通安全、電子銀行、AI/新興科技、供應鏈風險、作業韌性與內部控制。", "size": 20, "color": "263238"},
                {"text": "原始 PDF 保留在 01_原始PDF，後續 Markdown、clean/tree、leaf JSON 與 embedding 均由這批資料轉換或衍生。", "size": 18, "color": "455A64"},
            ])
            + footer(3)
        )
    )
    slides.append(
        slide_xml(
            rect(70, "Accent", 0, 0, 13.33, 0.13, "2C7A7B")
            + title_box("處理流程", "從文件轉換到檢索問答的端到端流程")
            + "".join(
                shape_text(
                    10 + i,
                    f"Step{i}",
                    emu(0.65 + i * 1.55),
                    emu(2.0),
                    emu(1.35),
                    emu(1.28),
                    [{"text": step, "size": 13, "bold": True, "color": "153E5C", "align": "c"}],
                    fill="F4F7F8" if i % 2 == 0 else "EEF6F5",
                    line="B8CDD2",
                    radius=True,
                )
                for i, step in enumerate(["原始 PDF", "PDF 轉 MD", "人工精修", "Clean Tree", "Leaf JSON", "Embedding", "問答 App", "展示驗證"])
            )
            + shape_text(30, "FlowNote", emu(0.9), emu(4.1), emu(11.5), emu(1.55), [
                {"text": "工作流以 01 到 08 的資料夾保存每個階段，讓生成區與精修區分開。重跑程式時可避免覆蓋人工修正，也方便追蹤每一層輸出的責任邊界。", "size": 20, "color": "263238"}
            ])
            + footer(4)
        )
    )
    slides.append(
        bullet_slide(
            5,
            "結構化策略",
            [
                "先將 PDF 粗轉 Markdown，再以人工精修提升段落、表格與標題品質。",
                "結構化程式產出 clean Markdown、headings、structure、tree 與 tables。",
                "leaf JSON 以法規階層切分，並可保留 ancestor context，使每個片段仍帶有來源脈絡。",
                "表格另外整理成 table cells 與 table chunks，避免重要欄位資訊被一般段落切段稀釋。",
            ],
        )
    )
    slides.append(
        bullet_slide(
            6,
            "Embedding 與比較設計",
            [
                "預設 embedding 模型為 BAAI/bge-m3。",
                "all_nodes：納入章、節、條、款、項等全部節點，適合廣泛召回。",
                "leaf：只納入最末端節點並帶上層脈絡，適合精準命中條文內容。",
                "table：聚焦表格 chunk，適合表格式規範、檢查項目與對照表查詢。",
                "hybrid：同時納入文字與表格，作為綜合型檢索模式。",
                "800200：固定 800 字、重疊 200 字的 baseline，用於比較結構化切段效果。",
            ],
        )
    )
    slides.append(
        slide_xml(
            rect(70, "Accent", 0, 0, 13.33, 0.13, "2C7A7B")
            + title_box("目前成果規模", "已完成可供檢索展示的 JSON 摘要與 embedding 資料")
            + shape_text(3, "M1", emu(0.8), emu(1.35), emu(2.25), emu(1.18), [{"text": f"{s['nodes']:,}", "size": 28, "bold": True, "color": "153E5C"}, {"text": "all nodes", "size": 12, "color": "455A64"}], fill="F4F7F8", line="D8E1E5", radius=True)
            + shape_text(4, "M2", emu(3.25), emu(1.35), emu(2.25), emu(1.18), [{"text": f"{s['leaves']:,}", "size": 28, "bold": True, "color": "153E5C"}, {"text": "leaf nodes", "size": 12, "color": "455A64"}], fill="F4F7F8", line="D8E1E5", radius=True)
            + shape_text(5, "M3", emu(5.7), emu(1.35), emu(2.25), emu(1.18), [{"text": f"{s['cells']:,}", "size": 28, "bold": True, "color": "153E5C"}, {"text": "table cells", "size": 12, "color": "455A64"}], fill="F4F7F8", line="D8E1E5", radius=True)
            + shape_text(6, "M4", emu(8.15), emu(1.35), emu(2.25), emu(1.18), [{"text": f"{s['chunks']:,}", "size": 28, "bold": True, "color": "153E5C"}, {"text": "table chunks", "size": 12, "color": "455A64"}], fill="F4F7F8", line="D8E1E5", radius=True)
            + shape_text(7, "M5", emu(10.6), emu(1.35), emu(2.25), emu(1.18), [{"text": "5", "size": 28, "bold": True, "color": "153E5C"}, {"text": "embedding modes", "size": 12, "color": "455A64"}], fill="F4F7F8", line="D8E1E5", radius=True)
            + shape_text(8, "ScaleNote", emu(0.9), emu(3.35), emu(11.5), emu(1.85), [
                {"text": "這些輸出已可支援業師展示：輸入問題後，系統可從不同 embedding 模式中召回相關條文或表格片段，並視需求接上 Ollama 本機模型產生回答。", "size": 20, "color": "263238"},
                {"text": "若只展示檢索結果，不需要啟動 LLM；若要展示完整回答，需先啟動 Ollama 與本機模型。", "size": 17, "color": "455A64"},
            ])
            + footer(7)
        )
    )
    slides.append(
        bullet_slide(
            8,
            "問答展示方式",
            [
                "App 可選擇不同 embedding 模式，觀察同一問題在不同切段策略下的召回差異。",
                "適合展示的問題類型：資安事件通報、電子銀行安全控管、供應鏈風險、AI 技術作業規範、內控制度檢核。",
                "展示時先看檢索片段與來源，再接 LLM 回答，能凸顯可追溯性與降低幻覺的設計。",
                "批次問答腳本可用於整理一批題目的答案，方便做評估表或比較結果。",
            ],
        )
    )
    slides.append(
        bullet_slide(
            9,
            "給業師可看的亮點",
            [
                "不是單純把 PDF 丟進向量資料庫，而是保留法規文件的階層與表格結構。",
                "結構化切段能讓答案更容易回到明確條文、章節與表格來源。",
                "同時保留固定字數 baseline，方便用實驗方式說明結構化方法的改善方向。",
                "資料流、精修區與生成區分離，後續要補資料或重跑 embedding 比較容易維護。",
            ],
        )
    )
    slides.append(
        bullet_slide(
            10,
            "後續工作",
            [
                "補齊代表性測試問題集，建立 retrieval hit rate、來源正確性與回答可用性的評估表。",
                "挑選 3 到 5 個業務情境做 demo，例如資安事件通報、AI 作業規範、電子銀行控管。",
                "整理人工精修規則，讓新增法規文件時能有一致的資料清理標準。",
                "若要對外展示，需確認原始法規資料來源、授權與引用規範。",
            ],
        )
    )
    return slides


def content_types(n: int) -> str:
    slides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, n + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  {slides}
</Types>"""


def presentation_xml(n: int) -> str:
    ids = "\n".join(f'<p:sldId id="{255+i}" r:id="rId{i}"/>' for i in range(1, n + 1))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{n+1}"/></p:sldMasterIdLst>
  <p:sldIdLst>{ids}</p:sldIdLst>
  <p:sldSz cx="{W}" cy="{H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


def presentation_rels(n: int) -> str:
    rels = "\n".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, n + 1)
    )
    rels += f'\n<Relationship Id="rId{n+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    rels += f'\n<Relationship Id="rId{n+2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>'
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>"""


MIN_MASTER = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""

MIN_LAYOUT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""

THEME = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Clean">
  <a:themeElements>
    <a:clrScheme name="Clean"><a:dk1><a:srgbClr val="222222"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="153E5C"/></a:dk2><a:lt2><a:srgbClr val="F4F7F8"/></a:lt2><a:accent1><a:srgbClr val="2C7A7B"/></a:accent1><a:accent2><a:srgbClr val="7AA095"/></a:accent2><a:accent3><a:srgbClr val="C54E4E"/></a:accent3><a:accent4><a:srgbClr val="E0B44F"/></a:accent4><a:accent5><a:srgbClr val="6E7F80"/></a:accent5><a:accent6><a:srgbClr val="455A64"/></a:accent6><a:hlink><a:srgbClr val="2C7A7B"/></a:hlink><a:folHlink><a:srgbClr val="5B6870"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="JhengHei"><a:majorFont><a:latin typeface="Microsoft JhengHei"/><a:ea typeface="Microsoft JhengHei"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft JhengHei"/><a:ea typeface="Microsoft JhengHei"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Clean"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>"""


def write_pptx() -> None:
    slides = build_slides()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(len(slides)))
        z.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>""")
        z.writestr("docProps/core.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>金融法規資料結構化與 RAG 問答專題</dc:title><dc:creator>Codex</dc:creator></cp:coreProperties>""")
        z.writestr("docProps/app.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex</Application><PresentationFormat>寬螢幕</PresentationFormat><Slides>{len(slides)}</Slides></Properties>""")
        z.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        z.writestr("ppt/slideMasters/slideMaster1.xml", MIN_MASTER)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>""")
        z.writestr("ppt/slideLayouts/slideLayout1.xml", MIN_LAYOUT)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>""")
        z.writestr("ppt/theme/theme1.xml", THEME)
        for i, slide in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide)
            z.writestr(
                f"ppt/slides/_rels/slide{i}.xml.rels",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>""",
            )


if __name__ == "__main__":
    write_pptx()
    print(OUT)
