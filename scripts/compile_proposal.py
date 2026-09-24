#!/usr/bin/env python3
"""
Script Kompilasi Proposal Penelitian Tugas Akhir
================================================
Menggabungkan seluruh berkas seksi modular dari 'docs/sections-proposal/'
menjadi satu berkas master utuh 'docs/proposal.md' secara rapi, berurutan,
dan dilengkapi dengan Daftar Isi (Table of Contents) otomatis.

CATATAN: Script ini HANYA MEMBACA berkas-berkas seksi dan TIDAK MENGHAPUS
atau mengubah berkas di 'docs/sections-proposal/'.
"""

import os
import re
import sys
import argparse
from pathlib import Path


def slugify(text: str) -> str:
    """Mengubah teks judul menjadi anchor link markdown github-style."""
    # Hapus karakter non-alphanumeric selain spasi dan tanda minus
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    # Ganti spasi dengan tanda minus
    text = re.sub(r"[\s_]+", "-", text)
    return text


def parse_section_number(filename: str) -> int:
    """Mengekstrak angka urutan dari nama berkas (misal: '10-apakah...' -> 10)."""
    match = re.match(r"^(\d+)", filename)
    if match:
        return int(match.group(1))
    return 9999


def clean_section_content(content: str, filename: str) -> tuple[str, str]:
    """
    Membersihkan dan menstandarisasi konten seksi agar memiliki hierarki heading yang rapi.
    Mengembalikan (title, cleaned_body).
    """
    lines = content.strip().splitlines()
    if not lines:
        return filename, ""

    title = ""
    start_idx = 0

    # Kasus khusus file 1: memuat banner 'Draft Proposal Penelitian...'
    if filename.startswith("1-") and "Draft Proposal Penelitian" in lines[0]:
        start_idx = 1
        while start_idx < len(lines) and not lines[start_idx].strip():
            start_idx += 1

    # Cari judul utama seksi
    for i in range(start_idx, min(len(lines), start_idx + 5)):
        line = lines[i].strip()
        if not line:
            continue
        # Pola '# 0. Judul' atau '1. Judul'
        clean_line = re.sub(r"^#+\s*", "", line).strip()
        if re.match(r"^\d+\.\s+", clean_line):
            title = clean_line
            start_idx = i + 1
            break

    if not title:
        # Fallback jika tidak ditemukan pola angka
        title = lines[0].strip().lstrip("#").strip()
        start_idx = 1

    remaining_lines = lines[start_idx:]
    
    # Standarisasi heading di dalam seksi:
    # Jika ada heading '# ...', ubah menjadi '### ...' agar berada di bawah '## Section'
    normalized_lines = []
    for line in remaining_lines:
        if line.startswith("# "):
            normalized_lines.append("### " + line[2:])
        elif line.startswith("## "):
            normalized_lines.append("### " + line[3:])
        else:
            normalized_lines.append(line)

    body = "\n".join(normalized_lines).strip()
    return title, body


def compile_proposal(sections_dir: Path, output_file: Path) -> None:
    print("=" * 70)
    print("  KOMPILASI DOKUMEN PROPOSAL PENELITIAN TUGAS AKHIR")
    print("=" * 70)
    print(f"Direktori sumber : {sections_dir.resolve()}")
    print(f"Berkas target    : {output_file.resolve()}")
    print("-" * 70)

    if not sections_dir.exists():
        print(f"ERROR: Direktori sumber {sections_dir} tidak ditemukan!")
        sys.exit(1)

    # Ambil seluruh berkas markdown di direktori seksi
    section_files = [f for f in sections_dir.iterdir() if f.is_file() and f.suffix.lower() == ".md"]
    if not section_files:
        print("ERROR: Tidak ada berkas markdown yang ditemukan di direktori seksi!")
        sys.exit(1)

    # Urutkan secara numerik berdasarkan prefiks angka (0, 1, 2, ..., 35)
    section_files.sort(key=lambda p: parse_section_number(p.name))

    print(f"Ditemukan {len(section_files)} berkas seksi modular.")
    print("-" * 70)

    parsed_sections = []
    toc_items = []

    for fpath in section_files:
        with open(fpath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        title, body = clean_section_content(raw_text, fpath.name)
        anchor = slugify(title)
        toc_items.append((title, anchor))
        parsed_sections.append({
            "filename": fpath.name,
            "title": title,
            "anchor": anchor,
            "body": body,
            "word_count": len(raw_text.split()),
            "line_count": len(raw_text.splitlines())
        })

    # Susun Dokumen Master Markdown
    doc_lines = []
    
    # 1. Header Utama
    doc_lines.append("# Proposal Penelitian Tugas Akhir")
    doc_lines.append("> **Learning Physical State Representations for Sensor-Grounded Reasoning and Future-State Prediction**  ")
    doc_lines.append("> *(Pemodelan Representasi Fisik Spasial-Temporal dan Penyelarasan LLM Berbasis Sinyal Radar mmWave)*\n")
    doc_lines.append("---\n")
    
    # 2. Metadata Proposal
    doc_lines.append("## Informasi Dokumen")
    doc_lines.append("- **Topik Penelitian:** Sensor Representation Learning, Physical Dynamics, and Cognitive Alignment")
    doc_lines.append("- **Modalitas Sensor:** mmWave Radar Point Cloud ($N=128$, fitur: $x, y, z, v_d, \\text{SNR}$)")
    doc_lines.append("- **Model Bahasa:** Small Language Model (Qwen2.5-1.5B-Instruct, **100% Strictly Frozen**)")
    doc_lines.append("- **Antarmuka Penyelaras:** Two-Layer MLP Projector (~1.97M parameter)")
    doc_lines.append("- **Dataset Empiris:** MM-Fi Dataset (Protocol 3, 40 subjek, 4 lingkungan, 27 aksi) + Pretrained Point-MAE ShapeNet")
    doc_lines.append(f"- **Metode Kompilasi:** Digabungkan secara otomatis dari {len(section_files)} berkas modular di [`docs/sections-proposal/`](sections-proposal/)\n")
    doc_lines.append("---\n")

    # 3. Daftar Isi Interaktif
    doc_lines.append("## Daftar Isi\n")
    for title, anchor in toc_items:
        doc_lines.append(f"- [{title}](#{anchor})")
    doc_lines.append("\n---\n")

    # 4. Konten Setiap Seksi
    print(f"{'No':<4} | {'Berkas Asal':<40} | {'Lines':<6} | {'Words':<6} | Judul Seksi")
    print("-" * 70)
    for idx, sec in enumerate(parsed_sections, start=1):
        print(f"{idx:<4} | {sec['filename']:<40} | {sec['line_count']:<6} | {sec['word_count']:<6} | {sec['title']}")
        
        doc_lines.append(f"## {sec['title']}\n")
        if sec["body"]:
            doc_lines.append(sec["body"])
        doc_lines.append("\n\n---\n")

    # 5. Penutup / Dokumen Footer
    doc_lines.append("## Penutup & Integritas Dokumen")
    doc_lines.append("Dokumen ini merupakan kompilasi terpadu dari seluruh rancangan seksi proposal penelitian. Seluruh parameter teknis, diagram arsitektur, dan protokol evaluasi yang termuat di dalamnya telah diselaraskan dengan implementasi kode aktif di repositori dan hasil pengujian empiris yang tervalidasi.\n")

    # Tulis ke berkas output
    compiled_text = "\n".join(doc_lines)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(compiled_text)

    total_lines = len(compiled_text.splitlines())
    total_words = len(compiled_text.split())
    file_size_kb = output_file.stat().st_size / 1024

    print("=" * 70)
    print("  KOMPILASI BERHASIL!")
    print(f"Berkas luaran : {output_file.resolve()}")
    print(f"Ukuran berkas : {file_size_kb:.2f} KB")
    print(f"Total baris   : {total_lines:,} baris")
    print(f"Total kata    : {total_words:,} kata")
    print(f"Status seksi  : Seluruh {len(section_files)} berkas di '{sections_dir.name}/' tetap AMAN dan UTUH.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Kompilasi berkas proposal modular menjadi berkas master utuh.")
    parser.add_argument(
        "--sections_dir",
        type=Path,
        default=Path("docs/sections-proposal"),
        help="Path ke direktori berkas seksi modular (default: docs/sections-proposal)"
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("docs/proposal.md"),
        help="Path ke berkas output markdown master (default: docs/proposal.md)"
    )
    args = parser.parse_args()
    compile_proposal(args.sections_dir, args.output_file)


if __name__ == "__main__":
    main()
