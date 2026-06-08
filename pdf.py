
import os
import json
import cv2
import numpy as np
from pdf2image import convert_from_path, pdfinfo_from_path
from tqdm import tqdm
import pytesseract

# =====================================================
# CONFIG
# =====================================================

PDF_FILE = "testLOKSATTA.pdf"

POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TXT_OUTPUT = os.path.join(OUTPUT_DIR, "loksatta_output.txt")
JSON_OUTPUT = os.path.join(OUTPUT_DIR, "loksatta_output.json")

# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess_image(page):
    img = np.array(page)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharp = cv2.filter2D(gray, -1, kernel)

    thresh = cv2.threshold(
        sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return thresh


# =====================================================
# PDF INFO
# =====================================================

info = pdfinfo_from_path(PDF_FILE, poppler_path=POPPLER_PATH)
total_pages = info["Pages"]
print(f"Total Pages: {total_pages}")

# =====================================================
# MAIN LOOP — all file writes happen inside this block
# =====================================================

all_pages = []

with open(TXT_OUTPUT, "w", encoding="utf-8") as txt_file:

    for page_no in range(1, total_pages + 1):

        print(f"Processing page {page_no}/{total_pages}")

        # ----- Convert PDF page to image -----
        page = convert_from_path(
            PDF_FILE,
            dpi=200,
            first_page=page_no,
            last_page=page_no,
            poppler_path=POPPLER_PATH
        )[0]

        processed = preprocess_image(page)

        # ----- Single OCR call per page -----
        data = pytesseract.image_to_data(
            processed,
            lang="mar+eng",
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT
        )

        # ----- Pass 1: build page-level text (line by line) -----
        line_map = {}
        blocks = {}

        for i in range(len(data["text"])):

            text = data["text"][i].strip()
            if not text:
                continue

            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = 0.0

            if conf < 20:
                continue

            block_num = data["block_num"][i]
            par_num   = data["par_num"][i]
            line_num  = data["line_num"][i]

            # For page-level ordered text
            line_key = (block_num, par_num, line_num)
            line_map.setdefault(line_key, []).append(text)

            # For block-wise article extraction
            if block_num not in blocks:
                blocks[block_num] = {
                    "top": data["top"][i],
                    "left": data["left"][i],
                    "lines": {}
                }
            block_lines = blocks[block_num]["lines"]
            block_lines.setdefault(line_num, []).append(text)

        # Build page text from ordered lines
        page_lines = [
            " ".join(line_map[k]) for k in sorted(line_map)
        ]
        page_text = "\n".join(page_lines)

        # ----- Write page-level text to TXT -----
        txt_file.write(f"\n{'='*100}\n")
        txt_file.write(f"PAGE {page_no}\n")
        txt_file.write(f"{'='*100}\n\n")
        txt_file.write(page_text)
        txt_file.write("\n\n")

        # ----- Pass 2: build articles from blocks -----
        articles = []

        for block_id, block_data in sorted(
            blocks.items(),
            key=lambda x: (x[1]["top"], x[1]["left"])
        ):
            assembled_lines = [
                " ".join(words)
                for _, words in sorted(block_data["lines"].items())
            ]

            article_text = "\n".join(assembled_lines).strip()

            if len(article_text) < 50:
                continue

            headline = assembled_lines[0] if assembled_lines else ""
            body = "\n".join(assembled_lines[1:]).strip()

            articles.append({
                "page": page_no,
                "block": block_id,
                "headline": headline,
                "body": body
            })

        # ----- Write block-wise articles to TXT -----
        txt_file.write(f"\n{'='*100}\n")
        txt_file.write(f"ARTICLES — PAGE {page_no}\n")
        txt_file.write(f"{'='*100}\n\n")

        for article in articles:
            txt_file.write(f"\n[BLOCK {article['block']}]\n")
            txt_file.write(article["headline"] + "\n\n")
            txt_file.write(article["body"] + "\n")

        # ----- Accumulate for JSON -----
        all_pages.extend(articles)

# =====================================================
# SAVE JSON  (after with block — txt_file already closed here)
# =====================================================

with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(all_pages, f, ensure_ascii=False, indent=2)

print("\nDONE")
print("TXT :", TXT_OUTPUT)
print("JSON:", JSON_OUTPUT)
