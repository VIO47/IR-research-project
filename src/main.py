import pyterrier as pt
from datasets import load_dataset
import os
import pandas as pd
from pyterrier_bert.bert4ir import *
import torch

def main():
    print(torch.cuda.is_available())
    pt.init()
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

    bm25 = setup_bm25(index_ref)
    bert_pipe = BERTPipeline(max_valid_rank=20)
    colbert = setup_colbert(documents_dicts)

    pipeline = bm25 >> colbert >> bert_pipe
    # Load a small set of queries for testing
    queries = dataset[["query_id", "query"]].drop_duplicates().head(100)
    queries.rename(columns={"query_id": "qid", "query": "query"}, inplace=True)

    # Run experiment
    experiment_results = run_experiment(pipeline, queries)
    print(experiment_results)

def setup_colbert(dataset):
    from pyterrier_colbert.ranking import ColBERTFactory
    from pyterrier_colbert.indexing import ColBERTIndexer
    colbert_index = ColBERTIndexer(checkpoint="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip", index_root=os.path.abspath("src/index"), index_name=os.path.abspath("src/colbert_index"), chunksize=3)
    colbert_index.index(dataset)
    colbert_factory = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip", os.path.abspath("src/colbert_index"), "colbert_index")
    colbert_reranker = colbert_factory.text_scorer()
    return colbert_reranker

def setup_bm25(index):
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    return bm25


def run_experiment(pipeline, queries):
    """
    Runs queries through the pipeline and evaluates retrieval performance.
    """
    print("Starting queries...")
    results = pipeline.transform(queries)

    # Ensure the results contain the necessary columns
    assert "qid" in results.columns and "docno" in results.columns and "score" in results.columns, "Results format incorrect!"

    # Evaluate performance (assuming qrels are available)
    qrels = load_qrels()
    evaluation = pt.Utils.evaluate(results, qrels, metrics=["nDCG@10", "MRR@10"])

    return evaluation


def load_qrels():
    """
    Loads the MS MARCO qrels for evaluation.
    """
    dataset = load_dataset("microsoft/ms_marco", 'v1.1', split="train[:1000]")  # Small subset
    qrels = dataset[["query_id", "doc_id", "relevance"]]
    qrels.rename(columns={"query_id": "qid", "doc_id": "docno", "relevance": "label"}, inplace=True)
    return qrels

if __name__ == "__main__":
    main()