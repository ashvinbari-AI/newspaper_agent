# compare_results.py
import os

# Read the output file
with open("output/loksatta_complete.txt", "r", encoding="utf-8") as f:
    extracted_text = f.read()

# Check for common improvements
print("="*60)
print("VERIFYING POST-PROCESSING IMPROVEMENTS")
print("="*60)

# Check 1: Count Marathi characters
marathi_chars = sum(1 for char in extracted_text if '\u0900' <= char <= '\u097F')
print(f"\n✅ Marathi characters found: {marathi_chars:,}")

# Check 2: Look for fixed patterns
patterns_to_check = {
    "Numbers converted": r'[०-९]',  # Devanagari numbers
    "Proper punctuation": r'[।॥]',   # Marathi danda
    "No stray '◌' characters": r'◌', # Combining character
}

for name, pattern in patterns_to_check.items():
    import re
    matches = re.findall(pattern, extracted_text)
    if name == "No stray '◌' characters":
        if len(matches) == 0:
            print(f"✅ {name}: No stray characters found (GOOD)")
        else:
            print(f"⚠️  {name}: {len(matches)} found")
    else:
        print(f"📊 {name}: {len(matches)} occurrences")

# Check 3: Sample text quality
print("\n" + "="*60)
print("SAMPLE EXTRACTED TEXT (First 500 chars):")
print("="*60)
print(extracted_text[:500])
print("\n... (truncated)")