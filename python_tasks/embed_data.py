import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/mnt/hf_cache"

PY3 = "/usr/bin/python3"
from pyspark.sql import SparkSession

spark = SparkSession.builder \
      .appName("Embed Cleaned Wiki Movie Data") \
      .config("spark.pyspark.python",       PY3) \
      .config("spark.pyspark.driver.python",PY3) \
      .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON",       PY3) \
      .config("spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON",PY3) \
      .config("spark.yarn.appMasterEnv.HF_HOME",              "/mnt/hf_cache") \
      .config("spark.yarn.appMasterEnv.SENTENCE_TRANSFORMERS_HOME", "/mnt/hf_cache") \
      .config("spark.executorEnv.PYSPARK_PYTHON",       PY3) \
      .config("spark.executorEnv.PYSPARK_DRIVER_PYTHON",PY3) \
      .config("spark.executorEnv.HF_HOME",              "/mnt/hf_cache") \
      .config("spark.executorEnv.TRANSFORMERS_CACHE",   "/mnt/hf_cache") \
      .config("spark.executorEnv.SENTENCE_TRANSFORMERS_HOME", "/mnt/hf_cache") \
      .getOrCreate()


from sentence_transformers import SentenceTransformer
        
input_path = "s3a://nina-rag-project/processed_data/movie_cleanedtexts"
raw_rdd = spark.sparkContext.textFile(input_path)
movie_rdd = raw_rdd.map(lambda line: line.split("\t", 1)).filter(lambda x: len(x) == 2)

# Sentence Embedding
def embed_partition(partition):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for key, text in partition:
        yield (key, model.encode(text).tolist())

embedded_rdd = movie_rdd.mapPartitions(embed_partition)

# Save to S3
output_path = "s3a://nina-rag-project/processed_data/movie_embeddings"

# Save as TSV format
embedded_rdd.map(lambda x: f"{x[0]}\t{','.join(map(str, x[1]))}") \
    .saveAsTextFile(output_path)

print(f"Saved embedded data to: {output_path}")
