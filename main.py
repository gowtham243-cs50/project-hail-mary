# filepath: /home/gowtham/COde-recet/main.py
import asyncio
from utilis import get_markdown_chunks
from db.chroma import index_pdfs

if __name__ == "__main__":
    asyncio.run(index_pdfs(source="/home/gowtham/Downloads/Ata.pdf",collection_name="unit-2"))
    print("Markdown chunks extracted successfully.")