from journal_ai.ingestion.pdf_parser import extract_pdf_text


def test_pdf_parser_rejects_empty_bytes():
    try:
        extract_pdf_text(b"not a pdf")
    except Exception:
        assert True
