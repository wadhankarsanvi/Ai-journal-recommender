import io
from pypdf import PdfReader


def extract_pdf_text(pdf_bytes: bytes, max_pages: int = 6) -> str:
    """
    Extracts text from the front matter of an academic manuscript PDF (default first 6 pages).
    Academic title, abstract, keywords, and methodology are located in the first few pages.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise ValueError(f"Could not read PDF format: {exc}")

    if not reader.pages:
        raise ValueError("The uploaded PDF has no readable pages.")

    pages = []
    # Only scan up to max_pages to keep response times fast and avoid memory spikes
    for page in reader.pages[:max_pages]:
        try:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text.strip())
        except Exception:
            continue

    text = "\n\n".join(pages).strip()

    if not text or len(text) < 20:
        raise ValueError(
            "No extractable text found in the PDF. It may be scanned/image-only."
        )

    return text

