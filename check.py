# check_all.py
from pyspark.sql import SparkSession

# Use the exact Python you installed into
PY3 = "/usr/bin/python3"

spark = SparkSession.builder \
    .appName("CheckAllBootstrapPackages") \
    .config("spark.pyspark.python",       PY3) \
    .config("spark.pyspark.driver.python",PY3) \
    .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON",       PY3) \
    .config("spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON",PY3) \
    .config("spark.executorEnv.PYSPARK_PYTHON",       PY3) \
    .config("spark.executorEnv.PYSPARK_DRIVER_PYTHON",PY3) \
    .getOrCreate()

sc = spark.sparkContext

modules = [
    "numpy",
    "rank_bm25",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "faiss",          # for faiss-cpu
    "nltk",
    "stanza",
    "typing_extensions",
]

def check_module(pkg):
    try:
        m = __import__(pkg)
        version = getattr(m, "__version__", "installed")
        return f"{pkg}: {version}"
    except Exception as e:
        return f"{pkg}: MISSING ({e.__class__.__name__})"

results = (
    sc.parallelize(modules, len(modules))
      .map(check_module)
      .collect()
)

print("Bootstrap package check results:")
for line in results:
    print("  ", line)

spark.stop()