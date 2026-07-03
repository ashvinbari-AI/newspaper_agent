import os
import json
import cv2
import numpy as np
from pdf2image import convert_from_path, pdfinfo_from_path
import pytesseract
import re
import sys
import concurrent.futures
from functools import partial

# Set UTF-8 encoding for console output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =====================================================
# CONFIG
# =====================================================

PDF_FILE = "Loksatta_Pune_20260615.pdf"
POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

JSON_OUTPUT = os.path.join(OUTPUT_DIR, "loksatta_complete.json")

# =====================================================
# TUNABLE PERFORMANCE SETTINGS
# =====================================================

DPI = 200                  # Reduced from 300 → ~55% fewer pixels, still great for OCR
MAX_WORKERS = 4            # Parallel OCR workers (set to os.cpu_count() for max speed)
BATCH_SIZE = 8             # Pages converted per pdf2image call (reduces subprocess overhead)
INTERMEDIATE_EVERY = 10    # Save intermediate JSON every N pages

# =====================================================
# IMAGE PROCESSING
# =====================================================

def pil_to_cv2(pil_image):
    np_image = np.array(pil_image)
    if len(np_image.shape) == 3 and np_image.shape[2] == 3:
        return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    return np_image


def preprocess_for_marathi(pil_image):
    img = pil_to_cv2(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

    # Otsu threshold
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# =====================================================
# OCR
# =====================================================

def ocr_marathi_page(pil_image):
    processed = preprocess_for_marathi(pil_image)
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'

    try:
        data = pytesseract.image_to_data(
            processed,
            lang='mar+eng',
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )

        text_blocks = []
        prev_line = -1
        current_line = []

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            try:
                conf = float(data['conf'][i])
            except (ValueError, TypeError):
                conf = 0

            if text and conf > 30:
                line_num = data['line_num'][i]
                if line_num != prev_line and prev_line != -1:
                    if current_line:
                        text_blocks.append(' '.join(current_line))
                        current_line = []
                current_line.append(text)
                prev_line = line_num

        if current_line:
            text_blocks.append(' '.join(current_line))

        extracted_text = clean_marathi_text('\n'.join(text_blocks))
        return extracted_text, 100 if extracted_text else 0

    except Exception as e:
        return "", 0


def clean_marathi_text(text):
    if not text:
        return text

    replacements = {
        '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
        '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
        '§': '', '€': '', '¢': '', '™': '', '®': '',
        'â': '', '€¢': '', 'Â': '', 'Ã': '', 'Å': '',
        '|': '', '•': '', '●': '', '○': '', '▪': '',
        '♦': '', '♥': '', '♠': '', '†': '', '‡': '',
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    text = re.sub(r'\s+', ' ', text)

    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not line.strip():
            continue
        devanagari_count = sum(1 for c in line if '\u0900' <= c <= '\u097F')
        if len(line) > 0 and (devanagari_count / len(line) > 0.1 or len(line) < 100):
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


# =====================================================
# WORKER: processes a single (page_no, pil_image) pair
# =====================================================

def process_single_page(args):
    """
    Called in a worker process. args = (page_no, pil_image)
    Returns a page_data dict.
    """
    page_no, pil_image = args
    try:
        text, confidence = ocr_marathi_page(pil_image)
        return {
            "page_number": page_no,
            "confidence": confidence,
            "text": text,
            "length": len(text)
        }
    except Exception as e:
        return {
            "page_number": page_no,
            "confidence": 0,
            "text": "",
            "length": 0,
            "error": str(e)
        }


# =====================================================
# MAIN EXTRACTION  (batched conversion + parallel OCR)
# =====================================================

def extract_pdf_fast(pdf_path, dpi=DPI, max_workers=MAX_WORKERS, batch_size=BATCH_SIZE):
    info = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_PATH)
    total_pages = info["Pages"]
    print(f"Total pages: {total_pages}  |  DPI: {dpi}  |  Workers: {max_workers}  |  Batch: {batch_size}")

    all_pages_data = [None] * total_pages   # pre-allocate so order is preserved
    processed_count = 0

    # ── Batch loop: convert BATCH_SIZE pages at once, then OCR them in parallel ──
    for batch_start in range(1, total_pages + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, total_pages)
        print(f"\n▶ Converting pages {batch_start}–{batch_end} ...", flush=True)

        try:
            page_images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=batch_start,
                last_page=batch_end,
                poppler_path=POPPLER_PATH,
                thread_count=2          # pdf2image internal thread count
            )
        except Exception as e:
            print(f"  Conversion error for batch {batch_start}-{batch_end}: {e}")
            for pn in range(batch_start, batch_end + 1):
                all_pages_data[pn - 1] = {
                    "page_number": pn,
                    "confidence": 0,
                    "text": "",
                    "length": 0,
                    "error": f"Conversion failed: {e}"
                }
            continue

        # Pair each image with its actual page number
        page_pairs = [
            (batch_start + i, img)
            for i, img in enumerate(page_images)
        ]

        print(f"  OCR on {len(page_pairs)} pages in parallel ...", flush=True)

        # ProcessPoolExecutor can't pickle PIL images easily across processes,
        # so use ThreadPoolExecutor (Tesseract releases the GIL during C calls).
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_page, pair): pair[0] for pair in page_pairs}
            for future in concurrent.futures.as_completed(futures):
                page_no = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "page_number": page_no,
                        "confidence": 0,
                        "text": "",
                        "length": 0,
                        "error": str(e)
                    }
                all_pages_data[page_no - 1] = result
                processed_count += 1

                # Progress
                preview = result['text'][:60].replace('\n', ' ') if result['text'] else "(empty)"
                print(f"  ✓ Page {page_no:>4}: {result['length']:>6} chars — {preview}", flush=True)

        # Save intermediate results
        completed = [p for p in all_pages_data if p is not None]
        if len(completed) % INTERMEDIATE_EVERY == 0 or batch_end == total_pages:
            _save_intermediate(completed, batch_end)

    return all_pages_data


# =====================================================
# SAVE
# =====================================================

def _save_intermediate(pages_data, up_to_page):
    temp = os.path.join(OUTPUT_DIR, f"temp_up_to_page_{up_to_page}.json")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)
    print(f"  💾 Intermediate JSON saved → {temp}", flush=True)


def save_final_json(pages_data):
    final_output = {
        "source_pdf": PDF_FILE,
        "total_pages": len(pages_data),
        "extraction_date": str(np.datetime64('today')),
        "pages": pages_data
    }
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Final JSON saved → {JSON_OUTPUT}")


def verify_results(pages_data):
    print("\n" + "=" * 60)
    print("EXTRACTION QUALITY REPORT")
    print("=" * 60)

    successful = [p for p in pages_data if p and p['length'] > 100]
    empty      = [p for p in pages_data if p and p['length'] == 0]
    total_chars = sum(p['length'] for p in successful)

    print(f"✅ Pages with text : {len(successful)}/{len(pages_data)}")
    print(f"⚠️  Empty pages     : {len(empty)}")
    print(f"📝 Total characters: {total_chars:,}")

    if successful:
        print("\n📄 Sample (first successful page):")
        print("-" * 60)
        print(successful[0]['text'][:500])
        print("-" * 60)


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MARATHI NEWSPAPER EXTRACTOR  —  FAST MODE")
    print("=" * 60)
    print(f"PDF       : {PDF_FILE}")
    print(f"Output    : {JSON_OUTPUT}")
    print()

    if not os.path.exists(PDF_FILE):
        print(f"❌ PDF not found: {PDF_FILE}")
        sys.exit(1)

    import time                          # ← ADD LINE 1 HERE
    start = time.time()                  # ← ADD LINE 2 HERE

    pages_data = extract_pdf_fast(PDF_FILE)

    save_final_json(pages_data)
    verify_results(pages_data)

    print(f"\n⏱ Total time: {(time.time() - start) / 60:.1f} minutes")   # ← ADD LINE 3 HERE

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\n📄 Your JSON file: {JSON_OUTPUT}")

# if __name__ == "__main__":
#     print("=" * 60)
#     print("MARATHI NEWSPAPER EXTRACTOR  —  FAST MODE")
#     print("=" * 60)
#     print(f"PDF       : {PDF_FILE}")
#     print(f"Output    : {JSON_OUTPUT}")
#     print(f"DPI       : {DPI}  (was 300)")
#     print(f"Workers   : {MAX_WORKERS}")
#     print(f"Batch size: {BATCH_SIZE} pages/conversion call")
#     print()

#     if not os.path.exists(PDF_FILE):
#         print(f"❌ PDF not found: {PDF_FILE}")
#         sys.exit(1)

#     pages_data = extract_pdf_fast(PDF_FILE)

#     save_final_json(pages_data)
#     verify_results(pages_data)

#     print("\n" + "=" * 60)
#     print("DONE!")
#     print("=" * 60)
#     print(f"\n📄 Your JSON file: {JSON_OUTPUT}")