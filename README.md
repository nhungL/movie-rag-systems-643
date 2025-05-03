1. **Setup**

- **Amazon EMR Cluster**  
  - EMR version: `emr-7.8.0`  
  - Installed applications: Hadoop 3.4.1, Hive 3.1.3, JupyterEnterpriseGateway 2.6.0, Livy 0.8.0, Spark 3.5.4  
  - Cluster configuration: 1 primary node, 3 core nodes, 0 task nodes
  - ⚠️ **Important:** Add the bootstrap script `bootstrap.sh` when creating the cluster to install required dependencies which are listed in `requirements.txt`.

- **S3 Storage**  
  - Full dataset: `s3://nina-rag-project/wiki_movie_plots_deduped.csv`  
    (original source: [Kaggle](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots))

2. **Steps to Run**

- **Task 1: Read dataset and prepare the data**
    ```
    spark-submit read_clean_data.py > res1.txt
    ```

- **Task 2: Generate embeddings for dense retrieval**
    ```
    spark-submit embed_data.py > res2.txt
    ```

- **Task 3: Get suggested movies (retrieval system)**
    ```
    spark-submit retrieval_system.py --query "Search query here" --topk 5 > res3.txt
    ```

- **Task 4: Generate natural language explanation**
    ```
    python3 generative_t5.py --retrieval_path retrieval_results.json > res4.txt
    ```

3. **Check Results**

- Review outputs in the `res*.txt` files (`res1.txt`, `res2.txt`, `res3.txt`, `res4.txt`) after each stage.
