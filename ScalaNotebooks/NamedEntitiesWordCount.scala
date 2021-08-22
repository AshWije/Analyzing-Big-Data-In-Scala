// DBTITLE 1,NAMED ENTITIES WORD COUNT
// Description:
//		Uses a pipeline to recognize named entities then counts the number of
//			times those specific entities are mentioned in a provided text
//			through map-reducing.

// COMMAND ----------

// DBTITLE 1,Import...
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._

SparkNLP.version()

// COMMAND ----------

// DBTITLE 1,Load...
// Pre-trained pipeline model
val recognize_entities_pipeline = PretrainedPipeline("recognize_entities_dl", "en")

// Text file
val data = spark.read.csv("/FileStore/tables/SherlockText.txt").toDF("text","id")

// Load list of stop words; obtained from https://countwordsfree.com/stopwords
val stopWords = spark.read.text("/FileStore/tables/StopWords.txt").collect().map(_.getString(0)).toSet

// COMMAND ----------

// DBTITLE 1,Annotate...
// Annotate data
val annotation = recognize_entities_pipeline.transform(data)

// Show all annotations
annotation.show()

// COMMAND ----------

// Show entities separated by commas
val entities = annotation.select("entities.result").select(concat_ws(",", $"result")).toDF("result")
entities.show(false)

// COMMAND ----------

// DBTITLE 1,Map...
// Get rdd, convert each row to a string, and convert to lowercase
val rdd1 = entities.rdd
              .map(_.getString(0))
              .map(_.toLowerCase)

// Remove all punctuation, symbols, empty strings, and stop words
val rdd2 = rdd1.map(_.replaceAll("[\\[\\]\"“”‘\\(\\),.!?:;_]|[—’']$", " ")
                     .replaceAll("  +", " ")
                     .trim).filter(_.length > 0)
                     .filter(!stopWords.contains(_))

// Map to (key, value) such that:
//     key = named entity
//     value = count
val rdd3 = rdd2.map(x => (x, 1))

// COMMAND ----------

// Display after mapping
rdd3.take(100)

// COMMAND ----------

// DBTITLE 1,Reduce...
// Reduce by key
val rdd4 = rdd3.reduceByKey((x, y) => x + y)

// Sort in descending order of counts
val rdd5 = rdd4.sortBy(-_._2)

// COMMAND ----------

// Display after reducing
rdd5.collect()