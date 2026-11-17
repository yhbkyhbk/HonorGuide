"""文档处理模块：攻略读写、主题分块与 JSON 保存。"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
import json
from pathlib import Path
import re
from typing import Iterable

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from config import ROOT as PROJECT_ROOT, DOCS_DIR as DEFAULT_DATA_DIR, STORAGE

EXCLUDED_SECTIONS = {"资料说明", "参考来源"}


MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+)\)")


def resolve_project_path(path: str | Path) -> Path:
    """相对路径统一相对于项目根目录，避免从不同目录启动时找不到资料。"""
    path = Path(path)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _json_date(value: object) -> str:
    """YAML 可将未加引号的日期解析为 date；保存时统一转换为 ISO 字符串。"""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"元数据包含不支持的类型：{type(value).__name__}")


def _read_frontmatter(text: str, source: Path) -> tuple[dict, str]:
    """把文件开头的 YAML 与 Markdown 正文分开。"""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{source.name}：缺少以 --- 开头的 YAML 元数据")

    closing_line = next(
        (i for i in range(1, len(lines)) if lines[i].strip() == "---"),
        None,
    )
    if closing_line is None:
        raise ValueError(f"{source.name}：YAML 元数据缺少结束标记 ---")

    try:
        metadata = yaml.safe_load("".join(lines[1:closing_line]))
    except yaml.YAMLError as exc:
        raise ValueError(f"{source.name}：YAML 格式错误：{exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{source.name}：YAML 元数据必须是键值映射")

    for key in ("doc_id", "hero"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"{source.name}：{key} 必须是非空字符串")
        metadata[key] = metadata[key].strip()

    try:
        metadata = json.loads(json.dumps(metadata, default=_json_date))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source.name}：元数据无法保存为 JSON：{exc}") from exc

    body = "".join(lines[closing_line + 1:]).strip()
    if not body:
        raise ValueError(f"{source.name}：Markdown 正文不能为空")
    return metadata, body


def load_documents(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[Document]:
    """读取目录下的攻略。每篇完整正文作为父文档，暂不删减任何小节。"""
    data_dir = resolve_project_path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"资料目录不存在：{data_dir}")
    files = sorted(data_dir.rglob("*.md"))
    if not files:
        raise ValueError(f"资料目录中没有 Markdown 文件：{data_dir}")

    documents = []
    seen_ids: dict[str, Path] = {}
    for file_path in files:
        text = file_path.read_text(encoding="utf-8-sig")
        metadata, body = _read_frontmatter(text, file_path)
        doc_id = metadata["doc_id"]
        if doc_id in seen_ids:
            raise ValueError(
                f"doc_id 重复：{doc_id}，涉及 {seen_ids[doc_id]} 和 {file_path}"
            )
        seen_ids[doc_id] = file_path
        source = (
            file_path.relative_to(PROJECT_ROOT).as_posix()
            if file_path.is_relative_to(PROJECT_ROOT)
            else file_path.as_posix()
        )
        metadata.update(
            source=source,
            parent_id=doc_id,
            doc_type="parent",
        )
        documents.append(Document(id=doc_id, page_content=body, metadata=metadata))
    return documents


def save_documents(path: Path, documents: list[Document]) -> None:
    records = [
        {
            "id": doc.id,
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in documents
    ]
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_parent_records() -> dict[str, dict]:
    rows = json.loads((STORAGE / "documents.json").read_text(encoding="utf-8"))
    return {row["metadata"]["doc_id"]: row for row in rows}


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[Document]:
    """先按标题分开主题，只对过长的小节继续递归切分。

    chunk_size 是字符上限；重叠只发生在同一个小节内，不跨技能或主题。
    完整父文档不被修改；子块排除资料说明、参考来源列表及只有标题的内容。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap 必须满足 0 <= chunk_overlap < chunk_size")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection"),
        ],
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        keep_separator="end",
    )

    all_chunks = []
    seen_ids: set[str] = set()
    for parent in documents:
        doc_id = parent.metadata["doc_id"]
        if doc_id in seen_ids:
            raise ValueError(f"doc_id 重复：{doc_id}")
        seen_ids.add(doc_id)
        chunk_index = 0

        for section_index, section in enumerate(
            header_splitter.split_text(parent.page_content)
        ):
            if section.metadata.get("section") in EXCLUDED_SECTIONS:
                continue
            content_without_headings = re.sub(
                r"(?m)^#{1,6}\s+.*$", "", section.page_content
            ).strip()
            if not content_without_headings:
                continue

            # 保存整个小节的链接；长小节切分后，各子块仍可找到完整来源 URL。
            source_urls = list(dict.fromkeys(MARKDOWN_LINK.findall(section.page_content)))
            section_metadata = {
                **parent.metadata,
                **section.metadata,
                "section": section.metadata.get("section", "正文"),
                "subsection": section.metadata.get("subsection", ""),
                "section_id": f"{doc_id}:section:{section_index:03d}",
                "section_index": section_index,
                "source_urls": source_urls,
            }
            section_doc = Document(
                page_content=section.page_content,
                metadata=section_metadata,
            )
            for section_chunk_index, chunk in enumerate(
                text_splitter.split_documents([section_doc])
            ):
                chunk_id = f"{doc_id}_{chunk_index:03d}"
                chunk.id = chunk_id
                chunk.metadata.update(
                    chunk_id=chunk_id,
                    parent_id=doc_id,
                    doc_type="child",
                    chunk_index=chunk_index,
                    section_chunk_index=section_chunk_index,
                )
                all_chunks.append(chunk)
                chunk_index += 1
    return all_chunks


def prepare_documents(chunk_size: int = 400, chunk_overlap: int = 50) -> dict:
    """供页面调用，处理当前攻略并保存完整父文档与子块。"""
    parents = load_documents()
    chunks = split_documents(parents, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("未生成任何有效分块，请检查文档内容")
    STORAGE.mkdir(parents=True, exist_ok=True)
    save_documents(STORAGE / "documents.json", parents)
    save_documents(STORAGE / "chunks.json", chunks)
    return {
        "documents": len(parents), "chunks": len(chunks),
        "by_hero": dict(Counter(chunk.metadata["hero"] for chunk in chunks)),
        "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
    }
