import pyterrier as pt
from sentence_transformers import SentenceTransformer

def main():
    # The newer version of msmarco is called 'msmarcov2_document'
    # But it contains 1M samples
    dataset = pt.get_dataset("msmarco_document")
    indexer = pt.IterDictIndexer("./index", overwrite=True)
    index_ref = indexer.index(dataset.get_corpus_iter())
    index = pt.IndexFactory.of(index_ref)

    print(dataset.get_corpus()[0])

    # Retrieve queries
    queries = dataset.get_topics()
    queries = queries.rename(columns={"qid": "qid", "query": "query"})

if __name__ == "__main__":
    main()