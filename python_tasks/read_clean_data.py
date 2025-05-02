import os
import re
from pyspark.sql import SparkSession

# Set JAVA_HOME
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

# Create Spark session
spark = SparkSession.builder \
    .appName("Read and Clean Wiki Movie Data") \
    .getOrCreate()

# Read from S3
df = spark.read.csv(
    "s3://nina-rag-project/wiki_movie_plots_deduped.csv",
    header=True, multiLine=True, escape="\"", quote="\""
)

print(f"Total number of rows: {df.count()}")

# Feature Engineering
def build_kv_pair(row):
    key = f"{row['Title']}_{row['Release Year']}"
    value = f"Movie: {row['Title']}\nOrigin: {row['Origin/Ethnicity']}\nGenre: {row['Genre']}\nDirector: {row['Director']}\nCast: {row['Cast']}\nPlot: {row['Plot']}"
    return (key, value)

movie_rdd = df.fillna("unknown") \
    .select("Title", "Release Year", "Origin/Ethnicity", "Genre", "Director", "Cast", "Plot") \
    .rdd.map(build_kv_pair)

# Cleaning
citation_pattern = re.compile(r'\[\d+\]')
url_pattern = re.compile(r'https?://\S+')
ws_pattern = re.compile(r'\s+')

def clean(data):
    key, text = data
    text = citation_pattern.sub('', text)
    text = text.replace("\\'", "'")
    text = url_pattern.sub('', text)
    text = ws_pattern.sub(' ', text).strip()
    return (key, text)

movie_rdd = movie_rdd.map(clean)

# Truncate
use_small_dataset = False
MIN_CHARS = 200 if use_small_dataset else 400
MAX_CHARS = 2500 if use_small_dataset else 3000

def truncate_and_filter_by_length(data):
    key, text = data
    text = text[:MAX_CHARS]
    if len(text) >= MIN_CHARS:
        return (key, text)
    else:
        return None

movie_rdd = movie_rdd.map(truncate_and_filter_by_length).filter(lambda x: x is not None).cache()

cleaned_count = movie_rdd.count()
total_count = df.count()

print(f"Cleaned count: {cleaned_count} rows left (keep {round(cleaned_count*100/total_count, 2)}%)")

# Save to S3
output_path = "s3://nina-rag-project/processed_data/movie_cleanedtexts"
movie_rdd.map(lambda x: f"{x[0]}\t{x[1]}").saveAsTextFile(output_path)
print(f"Saved cleaned texts to: {output_path}")
