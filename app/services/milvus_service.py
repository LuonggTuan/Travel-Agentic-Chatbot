from pymilvus import Collection
import os
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
import tempfile
from typing import List, Dict
from app.db.milvus import connect_milvus, check_collection_milvus
from app.llms.embedding_models import get_embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils import logger

def normalize_docx_to_chunks(file_bytes: bytes, file_name: str) -> List[Dict]:
    # Save bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Convert Docx
    converter = DocumentConverter()
    result = converter.convert(tmp_path).document

    # Chunk
    chunker = HierarchicalChunker()
    chunk_iter = chunker.chunk(dl_doc=result)

    data_list = []
    # Process chunks by docling metadata
    for i, chunk in enumerate(chunk_iter):
        headings = chunk.meta.headings or []
        data = {"index": None,"headings": None, "type": None, "content": None}
        data["index"] = i
        data["headings"] = headings
        first_label = str(getattr(chunk.meta.doc_items[0], "label", "")).lower()

        data["type"] = first_label
        if first_label == "table":
            table_item = chunk.meta.doc_items[0]
            df = table_item.export_to_dataframe()
            markdown_table = df.to_markdown(index=False)
            data["content"] = markdown_table
        elif first_label == "list_item":
            list_item = chunk.text
            data["content"] =  list_item
        elif first_label == "paragraph": 
            paragraph = chunk.text
            data["content"] = paragraph
        elif first_label == "text": 
            text = chunk.text
            data["content"] = text
        data_list.append(data)
    
    os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Split further the content except TABLE
    final_data_list = []
    for data in data_list:
        index = data["index"]
        type_ = data["type"]
        headings = data["headings"]
        content = data["content"]
        
        # Không split TABLE
        if type_ == "table":
            final_data_list.append({
                "id": f"{index}",
                "type": type_,
                "headings": headings,
                "content": content
            })
            continue
        sub_chunks = splitter.split_text(content)
        # Heading text
        heading_text = " > ".join(h.strip() for h in headings if h.strip()) if headings else ""
        for j, sub in enumerate(sub_chunks):
            final_data_list.append({
                "id": f"{index}_{j}",
                "type": type_,
                "headings": headings,
                "content": heading_text + "\n" + sub
            })
    return final_data_list

def upload_chunks_to_milvus(data_list: List[Dict], collection_name: str):
    check_collection_milvus(collection_name)

    embedding_model = get_embedding_model()
    collection = Collection(name=collection_name)
    collection.load()

    texts = [item["content"] for item in data_list]
    vectors = embedding_model.embed_documents(texts)

    ids = []
    headings = []
    types = []
    contents = []
    all_vectors = []

    for i, (item, vec) in enumerate(zip(data_list, vectors)):
        ids.append(i)
        headings.append(" > ".join(item.get("headings", [])) if item.get("headings") else "")
        types.append(item.get("type", ""))
        contents.append(item.get("content", ""))
        all_vectors.append(vec)

    result = collection.insert([
        ids,
        headings,
        types,
        contents,
        all_vectors
    ])

    collection.flush()
    logger.info(f"Inserted {len(ids)} vectors into Milvus collection '{collection_name}'.")
    return result

def query_milvus(collection_name: str, query: str, top_k: int = 3):
    collection = Collection(name=collection_name)
    collection.load()

    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(query)

    search_params = {
        "metric_type": "COSINE"
    }

    results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["id", "heading", "type", "content"]
        )

    hits = []
    for hit in results[0]:
        hits.append({
            "id": hit.id,
            "score": hit.distance,
            "heading": hit.entity.get("heading"),
            "type": hit.entity.get("type"),
            "content": hit.entity.get("content"),
        })

    return hits