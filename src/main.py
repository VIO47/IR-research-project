import pyterrier as pt
from datasets import load_dataset
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

def main():
    # The newer version of msmarco is called 'msmarcov2_document'
    # But it contains 1M samples
    dataset = load_dataset("microsoft/ms_marco", 'v1.1', split="train[:10000]")  # Load a small subset to test setup
    documents = dataset.to_pandas()
    documents["query_id"] = documents["query_id"].astype(str)
    documents.rename(columns={"query_id": "docno", "query": "text"}, inplace=True)

    documents_dicts = documents.to_dict(orient="records")

    index_path = os.path.abspath("src/index")
    indexer = pt.IterDictIndexer(
        index_path=index_path,
        meta = {
            "docno": 20,
            "text": 131072
        },
        stemmer="porter",
        stopwords="terrier"
    )
    index_ref = indexer.index(documents_dicts)
    tf_idf = pt.terrier.Retriever(
        index_ref, wmodel="TF_IDF", num_results=5, metadata=["docno", "text"]
    )
    print("Indexing completed successfully!")

if __name__ == "__main__":
    main()