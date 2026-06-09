import os
import json
import cv2
import numpy as np
from pdf2image import convert_from_path, pdfinfo_from_path
from tqdm import tqdm
import pytesseract
import re
from collections import OrderedDict 
import unicodedata

# =====================================================
# CONFIG
# =====================================================

PDF_FILE = "Loksatta_Nagpur_20260608.pdf"

POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TXT_OUTPUT = os.path.join(OUTPUT_DIR, "loksatta_complete.txt")
JSON_OUTPUT = os.path.join(OUTPUT_DIR, "loksatta_complete.json")
LAYOUT_OUTPUT = os.path.join(OUTPUT_DIR, "layout_analysis.json")

# =====================================================
# ENHANCED PREPROCESSING (Multiple strategies)
# =====================================================

def preprocess_image_adaptive(page, strategy='aggressive'):
    """
    Multiple preprocessing strategies for best OCR results
    """
    img = np.array(page)
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    if strategy == 'basic':
        # Basic thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    elif strategy == 'aggressive':
        # Denoise first
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Sharpening
        kernel_sharpen = np.array([[-1,-1,-1],
                                    [-1, 9,-1],
                                    [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # Binary thresholding
        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to connect broken characters
        kernel = np.ones((2,2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return thresh
    
    elif strategy == 'light':
        # Light preprocessing for pages that are already good
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return thresh
    
    else:
        return gray

def detect_layout_region(image):
    """
    Detect different regions: text blocks, headers, columns
    """
    # Dilate to find text regions
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter small regions
        if w > 50 and h > 20:
            regions.append({
                'x': x, 'y': y, 'width': w, 'height': h,
                'area': w * h
            })
    
    # Sort by y position (top to bottom)
    regions.sort(key=lambda r: (r['y'], r['x']))
    
    return regions

# =====================================================
# ADVANCED MARATHI POST-PROCESSING
# =====================================================

def advanced_marathi_postprocessing(text):
    """
    Advanced fixes for remaining Marathi OCR issues
    """
    if not text:
        return text
    
    # Fix common Marathi character confusions
    marathi_fixes = {
        # Vowel signs (matras)
        'ि◌': 'ि', 'ी◌': 'ी', 'ु◌': 'ु', 'ू◌': 'ू',
        'े◌': 'े', 'ै◌': 'ै', 'ो◌': 'ो', 'ौ◌': 'ौ',
        
        # Consonant fixes
        'क◌': 'क', 'ख◌': 'ख', 'ग◌': 'ग', 'घ◌': 'घ',
        'च◌': 'च', 'छ◌': 'छ', 'ज◌': 'ज', 'झ◌': 'झ',
        'ट◌': 'ट', 'ठ◌': 'ठ', 'ड◌': 'ड', 'ढ◌': 'ढ',
        'त◌': 'त', 'थ◌': 'थ', 'द◌': 'द', 'ध◌': 'ध',
        'न◌': 'न', 'प◌': 'प', 'फ◌': 'फ', 'ब◌': 'ब',
        'भ◌': 'भ', 'म◌': 'म', 'य◌': 'य', 'र◌': 'र',
        'ल◌': 'ल', 'व◌': 'व', 'श◌': 'श', 'ष◌': 'ष',
        'स◌': 'स', 'ह◌': 'ह', 'ळ◌': 'ळ', 'क्ष◌': 'क्ष',
        'ज्ञ◌': 'ज्ञ', 'त्र◌': 'त्र',
        
        # Common number confusions
        '०': '०', '१': '१', '२': '२', '३': '३',
        '४': '४', '५': '५', '६': '६', '७': '७',
        '८': '८', '९': '९',
        
        # Punctuation
        '।': '।', '॥': '॥',
        
        # English to Marathi number conversions
        '0': '०', '1': '१', '2': '२', '3': '३',
        '4': '४', '5': '५', '6': '६', '7': '७',
        '8': '८', '9': '९',
    }
    
    for wrong, correct in marathi_fixes.items():
        text = text.replace(wrong, correct)
    
    # Normalize Unicode to composed form
    text = unicodedata.normalize('NFC', text)
    
    # Fix line breaks within words
    text = re.sub(r'(\w+)[-\s]+\n\s*(\w+)', r'\1\2', text)
    
    # Remove stray characters (keep Marathi, English, numbers, basic punctuation)
    text = re.sub(r'[^\u0900-\u097F\s\w\.\,\!\?\;\"\'\:\|\(\)\[\]\{\}0-9]', '', text)
    
    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix line spacing
    text = re.sub(r' \n ', '\n', text)
    
    # Remove empty lines at start/end
    text = text.strip()
    
    return text

# =====================================================
# OCR WITH LAYOUT
# =====================================================

def ocr_with_layout(image, lang="mar+eng"):
    """
    OCR with layout preservation and sequence maintenance
    """
    height, width = image.shape
    
    # Try multiple PSM modes and pick best
    psm_modes = [6, 4, 3, 1]  # 6=block, 4=column, 3=auto, 1=auto
    
    best_data = None
    best_confidence = 0
    
    for psm in psm_modes:
        try:
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                config=f"--oem 3 --psm {psm}",
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence
            confs = [float(c) for c in data['conf'] if float(c) > 0]
            avg_conf = np.mean(confs) if confs else 0
            
            if avg_conf > best_confidence:
                best_confidence = avg_conf
                best_data = data
                
        except Exception as e:
            continue
    
    return best_data, best_confidence

# =====================================================
# RECONSTRUCT TEXT SEQUENTIAL (SINGLE VERSION WITH POST-PROCESSING)
# =====================================================

def reconstruct_text_sequential(data, min_confidence=30):
    """
    Reconstruct text preserving exact reading order
    using line_num, word_num, and spatial positioning
    """
    if not data:
        return "", []
    
    # Group by line number and block
    lines = {}
    words_info = []
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if not text:
            continue
        
        try:
            conf = float(data['conf'][i])
        except (ValueError, TypeError):
            conf = 0.0
        
        if conf < min_confidence:
            continue
        
        block_num = data['block_num'][i]
        line_num = data['line_num'][i]
        word_num = data['word_num'][i]
        left = data['left'][i]
        top = data['top'][i]
        
        # Store word info for spatial analysis
        words_info.append({
            'text': text,
            'block': block_num,
            'line': line_num,
            'word': word_num,
            'left': left,
            'top': top,
            'conf': conf
        })
        
        # Group by block and line
        key = (block_num, line_num)
        if key not in lines:
            lines[key] = []
        lines[key].append((word_num, text))
    
    # Sort within each line by word_num
    for key in lines:
        lines[key].sort(key=lambda x: x[0])
    
    # Sort lines by block, then line_num
    sorted_keys = sorted(lines.keys(), key=lambda x: (x[0], x[1]))
    
    # Build sequential text
    sequential_text = []
    current_block = None
    
    for block, line in sorted_keys:
        if current_block != block:
            if current_block is not None:
                sequential_text.append("\n")  # Block separator
            current_block = block
        
        line_text = " ".join([word for _, word in lines[(block, line)]])
        sequential_text.append(line_text)
    
    raw_text = "\n".join(sequential_text)
    
    # =====================================================
    # APPLY ADVANCED MARATHI POST-PROCESSING HERE
    # =====================================================
    processed_text = advanced_marathi_postprocessing(raw_text)
    
    return processed_text, words_info

# =====================================================
# COLUMN-AWARE TEXT EXTRACTION
# =====================================================

def extract_columns_sequential(image, lang="mar+eng"):
    """
    Detect columns and extract text in correct reading order
    """
    height, width = image.shape
    
    # Simple column detection (assume 2 columns for newspaper)
    col_width = width // 2
    col1 = image[:, :col_width]
    col2 = image[:, col_width:]
    
    all_text = []
    
    # Process left column
    data1, conf1 = ocr_with_layout(col1, lang)
    if data1:
        text1, _ = reconstruct_text_sequential(data1)
        if text1:
            all_text.append(f"[COLUMN 1]\n{text1}")
    
    # Process right column
    data2, conf2 = ocr_with_layout(col2, lang)
    if data2:
        text2, _ = reconstruct_text_sequential(data2)
        if text2:
            all_text.append(f"[COLUMN 2]\n{text2}")
    
    return "\n\n".join(all_text)

# =====================================================
# MAIN EXTRACTION LOOP
# =====================================================

def extract_pdf_complete(pdf_path):
    """
    Complete extraction with all improvements
    """
    # Get PDF info
    info = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_PATH)
    total_pages = info["Pages"]
    print(f"Total pages to process: {total_pages}")
    
    all_content = []
    all_pages_data = []
    
    with open(TXT_OUTPUT, "w", encoding="utf-8") as txt_file:
        
        for page_no in tqdm(range(1, total_pages + 1), desc="Processing pages"):
            
            # Convert page to image
            page_image = convert_from_path(
                pdf_path,
                dpi=500,  # Higher DPI for better OCR
                first_page=page_no,
                last_page=page_no,
                poppler_path=POPPLER_PATH
            )[0]
            
            # Try different preprocessing strategies
            strategies = ['aggressive', 'basic', 'light']
            best_text = ""
            best_confidence = 0
            
            for strategy in strategies:
                try:
                    # Preprocess
                    processed = preprocess_image_adaptive(page_image, strategy)
                    
                    # OCR with layout detection
                    ocr_data, confidence = ocr_with_layout(processed)
                    
                    if confidence > best_confidence and ocr_data:
                        best_confidence = confidence
                        # reconstruct_text_sequential now includes advanced post-processing
                        sequential_text, word_info = reconstruct_text_sequential(ocr_data)
                        
                        # Also try column-based extraction if standard fails
                        if len(sequential_text) < 100:
                            column_text = extract_columns_sequential(processed)
                            if len(column_text) > len(sequential_text):
                                sequential_text = column_text
                                # Apply post-processing to column text as well
                                sequential_text = advanced_marathi_postprocessing(sequential_text)
                        
                        best_text = sequential_text
                    
                    # Save intermediate processed image for debugging
                    if confidence > 70:
                        debug_path = os.path.join(OUTPUT_DIR, f"page_{page_no}_processed.png")
                        cv2.imwrite(debug_path, processed)
                        
                except Exception as e:
                    print(f"  Strategy {strategy} failed: {e}")
                    continue
            
            # Final cleanup (ensure post-processing is applied)
            if best_text:
                # One final pass of advanced post-processing
                best_text = advanced_marathi_postprocessing(best_text)
            
            # Write to file with clear page markers
            txt_file.write(f"\n{'='*100}\n")
            txt_file.write(f"PAGE {page_no} (Confidence: {best_confidence:.1f}%)\n")
            txt_file.write(f"{'='*100}\n\n")
            txt_file.write(best_text)
            txt_file.write("\n\n")
            
            # Store for JSON
            page_data = {
                "page_number": page_no,
                "confidence": best_confidence,
                "text": best_text,
                "length": len(best_text)
            }
            all_pages_data.append(page_data)
            
            # For content array
            if best_text.strip():
                all_content.append({
                    "page": page_no,
                    "content": best_text
                })
    
    return all_content, all_pages_data

# =====================================================
# VERIFICATION AND VALIDATION
# =====================================================

def verify_extraction_quality(all_pages_data):
    """
    Verify extraction quality and identify problem pages
    """
    print("\n" + "="*60)
    print("EXTRACTION QUALITY REPORT")
    print("="*60)
    
    total_chars = sum(p['length'] for p in all_pages_data)
    if all_pages_data:
        avg_confidence = np.mean([p['confidence'] for p in all_pages_data])
    else:
        avg_confidence = 0
    
    print(f"Total characters extracted: {total_chars:,}")
    print(f"Average confidence: {avg_confidence:.1f}%")
    print(f"Total pages processed: {len(all_pages_data)}")
    
    # Pages with low confidence
    low_conf_pages = [p for p in all_pages_data if p['confidence'] < 50]
    if low_conf_pages:
        print(f"\n⚠️  Low confidence pages (<50%):")
        for p in low_conf_pages:
            print(f"   Page {p['page_number']}: {p['confidence']:.1f}% confidence")
    
    # Empty pages
    empty_pages = [p for p in all_pages_data if p['length'] < 100]
    if empty_pages:
        print(f"\n⚠️  Pages with minimal text (<100 chars):")
        for p in empty_pages:
            print(f"   Page {p['page_number']}: {p['length']} chars")
    
    return avg_confidence

# =====================================================
# RUN COMPLETE EXTRACTION
# =====================================================

if __name__ == "__main__":
    
    print("="*60)
    print("COMPLETE PDF TEXT EXTRACTION SYSTEM")
    print("="*60)
    print(f"PDF: {PDF_FILE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()
    
    # Extract all content
    all_content, all_pages_data = extract_pdf_complete(PDF_FILE)
    
    # Save JSON with full structure
    final_output = {
        "source_pdf": PDF_FILE,
        "total_pages": len(all_pages_data),
        "extraction_date": str(np.datetime64('today')),
        "pages": all_pages_data,
        "full_content": all_content
    }
    
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    # Save layout analysis
    layout_info = {
        "pages_processed": len(all_pages_data),
        "average_confidence": float(np.mean([p['confidence'] for p in all_pages_data])) if all_pages_data else 0,
        "total_text_length": sum(p['length'] for p in all_pages_data)
    }
    
    with open(LAYOUT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(layout_info, f, ensure_ascii=False, indent=2)
    
    # Verify quality
    verify_extraction_quality(all_pages_data)
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    print(f"✅ Text file: {TXT_OUTPUT}")
    print(f"✅ JSON file: {JSON_OUTPUT}")
    print(f"✅ Layout analysis: {LAYOUT_OUTPUT}")
    print("\n💡 Tip: For manual verification, check the processed page images in output folder")