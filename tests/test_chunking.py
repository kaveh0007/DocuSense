from backend.pdf_utils import smart_chunk_pages, PageText

def test_chunking_basic():
    pages = [
        PageText(page_num=1, text="Para1\n\nPara2\n\nPara3 " * 10),
    ]
    chunks = smart_chunk_pages(pages, chunk_chars=200, overlap=50)
    assert len(chunks) >= 1
    assert all("text" in c and "page" in c for c in chunks)
