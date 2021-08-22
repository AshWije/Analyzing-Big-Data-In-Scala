// DBTITLE 1,MOVIE SEARCH ENGINE
// Description:
//		Create a search engine for single word and multiple word
//			searches on a provided set of movie plot summaries
//			by utilizing TF-IDF and cosine similarity.

// COMMAND ----------

// DBTITLE 1,Import...
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import scala.math


SparkNLP.version()

// COMMAND ----------

// DBTITLE 1,Load...
// Load plot summaries
val plot_data = spark.read.text("/FileStore/tables/plot_summaries.txt")

// Load movie metadata
val movie_metadata = spark.read.option("sep", "\t").csv("/FileStore/tables/movie_metadata.tsv")
        .toDF("id", "freebase_id", "name", "release_date", "box_office_revenue", "runtime", "languages", "countries", "genres")

// Extract only the movie id and name from the metadata
val movie_names = movie_metadata.select("id", "name")

// Load user search terms
val search_terms = spark.read.text("/FileStore/tables/SearchTerms.txt")

// Load list of stop words; obtained from https://countwordsfree.com/stopwords
val stopWords = spark.read.text("/FileStore/tables/StopWords.txt").collect().map(_.getString(0))

// COMMAND ----------

// DBTITLE 1,Annotate...
// Create document assembler
val documentAssembler = new DocumentAssembler()
       .setInputCol("value")
       .setOutputCol("document")

// Create sentence detector
val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

// Create tokenizer
val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

// Create normalizer
val normalizer = new Normalizer()
    .setInputCols(Array("token"))
    .setOutputCol("normalized")
    .setCleanupPatterns(Array("[^a-zA-Z0-9]"))
    .setLowercase(true)

// Create stemmer (can be used for better results)
//val stemmer = new Stemmer()
//    .setInputCols(Array("normalized"))
//    .setOutputCol("stem")

// Create stop words cleaner
val stopWordsCleaner = new StopWordsCleaner()
      .setInputCols("normalized")
      .setOutputCol("cleanTokens")
      .setStopWords(stopWords)
      .setCaseSensitive(false)

// Create finisher
val finisher = new Finisher()
      .setInputCols("cleanTokens")
      .setIncludeMetadata(false)

// Create pipeline
val pipeline = new Pipeline()
      .setStages(Array(
          documentAssembler,
          sentenceDetector,
          tokenizer,
          normalizer,
          //stemmer,
          stopWordsCleaner,
          finisher
      ))

// Annotate using pipeline
val annotation = pipeline.fit(plot_data).transform(plot_data)

// COMMAND ----------

// DBTITLE 0,Create Dataframe...
// Create dataframe
val df1 = annotation.select("finished_cleanTokens")
        .toDF("doc")
        .select(concat_ws(",", $"doc"))

df1.show()

// COMMAND ----------

// DBTITLE 1,Determine Term Frequency (TF)...
// Create the rdd from the dataframe
val rdd1 = df1.rdd                     // Turn df into rdd
        .map(_.getString(0))           // Turn every row into a string
        .map(_.replaceFirst(",", ":")) // Separate id from plot summary with ':' i.e. "id:plotSummaryString"
        .map(_.split(":"))             // Create array with two elements: id and plotSummaryString
        .flatMap(x => {                // Split plotSummaryString into tokens and let rdd be of the form (id, token)
            val id = x(0)
            val doc = x(1)
            val tokens = doc.split(",")
            tokens.map(token => (id, token))
        })
rdd1.take(100)

// COMMAND ----------

// Mapping:
//    key   = (id, term)
//    value = 1
val rdd2 = rdd1.map{case(id, term) => ((id, term), 1)}
rdd2.take(100)

// COMMAND ----------

// Reducing: Determine term frequency for each document
//     ((id, term), tf)
val tfRDD = rdd2.reduceByKey((x, y) => x + y)
tfRDD.take(100)

// COMMAND ----------

// DBTITLE 1,Determine Document Frequency (DF)...
// Mapping:
//    key   = term
//    value = 1
val rdd4 = tfRDD.map{case((id, term), tf) => (term, 1)}
rdd4.take(100)

// COMMAND ----------

// Reduce: Determine document frequency for each term
//    (term, df)
val dfRDD = rdd4.reduceByKey((x, y) => x + y)
rdd4.take(100)

// COMMAND ----------

// DBTITLE 1,Determine Inverse Document Frequency (IDF)...
// Get movie/document count
val numMovies = df1.count().toDouble

// Formula for idf:
//    idf = log (N / df)
// Map to (term, idf)
val idfRDD = dfRDD.map{case(term, df) => (term, math.log(numMovies/df.toDouble))}
idfRDD.take(100)

// COMMAND ----------

// DBTITLE 1,Determine TF-IDF...
// Alter tfRDD to be of the form (term, (id, tf))
val alteredTfRDD = tfRDD.map{case((id, term), tf) => (term, (id, tf))}

// COMMAND ----------

// Join:
//    alteredTfRDD -> (term, (id, tf))
//    idfRDD       -> (term, idf)
// Creates (term, ((id, tf), idf))
val joined = alteredTfRDD.join(idfRDD)
joined.take(100)

// COMMAND ----------

// Formula for tf-idf:
//    tf-idf = tf * idf
// Map to ((term, id), tfIdf)
val tfIdfRDD = joined.map{case(term, ((id, tf), idf)) => ((term, id), tf.toDouble*idf)}
tfIdfRDD.take(100)

// COMMAND ----------

val tfIdfDF = tfIdfRDD.map{case((term, id), tfIdf) => (term, id, tfIdf)}
          .toDF("term", "id", "tf-idf")
tfIdfDF.show()

// COMMAND ----------

// DBTITLE 1,Setup for Searching...
// Annotate search terms
val annotatedSearchTerms = pipeline.fit(search_terms).transform(search_terms)
annotatedSearchTerms.show(false)

// COMMAND ----------

// Create dataframe
val termsDF = annotatedSearchTerms
        .withColumn("clean_tokens", concat_ws(",", $"finished_cleanTokens"))
        .select("value", "clean_tokens")
termsDF.show(false)

// COMMAND ----------

// Create rdd for search terms
val searchTerms = termsDF.rdd                             // Turn df into rdd
        .map(row => (row.getString(0), row.getString(1))) // Turn every row into a pair (original terms, cleaned terms)
searchTerms.collect()

// COMMAND ----------

// Join tfIdfDF with movie_names on id
val tfIdfDFWithNames = tfIdfDF.join(movie_names, Seq("id"))
tfIdfDFWithNames.show()

// COMMAND ----------

// Convert idf RDD to DF
val idfDF = idfRDD.toDF("term", "idf")
idfDF.show()

// COMMAND ----------

// Calculate |doc| for each document
val docSizeDF = tfIdfDFWithNames.withColumn("square", $"tf-idf" * $"tf-idf").groupBy("id").agg(sum("square") as "size")
docSizeDF.show()

// COMMAND ----------

// DBTITLE 1,Search for Single Word Search Terms...
// Split search terms into only single word search terms
val singleWordSearchTerms = searchTerms.filter{case(origLine, cleanLine) => cleanLine.split("""\W""").length == 1}.collect()

// COMMAND ----------

// Loop through single word search terms
singleWordSearchTerms.foreach{case(origT, t) => {
  
    // Print message about term being searched
    println("Searching for " + origT + "...")
  
    // Get only movies with term in plot summary
    val termDF = tfIdfDFWithNames.filter($"term" === t)

    // Get top movies with the highest tf-idf values for that term
    val topMoviesTerm = termDF.orderBy(desc("tf-idf"))
  
    // Show only top ten movie names
    topMoviesTerm.select("name").show(10, false)
}}

// COMMAND ----------

// DBTITLE 1,Search for Multiple Word Search Terms...
// Split search terms into only multiple word search terms
val multiWordSearchTerms = searchTerms.filter{case(origLine, cleanLine) => cleanLine.split("""\W""").length > 1}.collect()

// COMMAND ----------

// Loop through lines of search terms
multiWordSearchTerms.foreach{case(origT, t) => {
  
    // Print message about term being searched
    println("Searching for " + origT + "...")
    
    // Split multiple word search term to single words
    val tList = t.split(",")
    
    // Create df using list of search terms
    val queryDF = sqlContext.sparkContext.parallelize(tList).toDF("term")
  
    // Calculate tf-idf for query
    //    Calculate tf
    val queryTfDF = queryDF.groupBy("term").agg(count("term") as "tf")
    
    //    Join with idfDF
    val queryJoinDF = queryTfDF.join(idfDF, Seq("term"))
  
    //    Calculate tf-idf
    val queryTfIdfDF = queryJoinDF.withColumn("tf-idf_query", $"tf" * $"idf")
                                  .select("term", "tf-idf_query")
  
    // Calculate |query|
    val querySize = queryTfIdfDF.withColumn("square", $"tf-idf_query" * $"tf-idf_query")
                                .agg(sum("square"))
                                .first()
                                .getDouble(0)
    
    // Calculate dot product of query and document
    val dotProdDF = queryTfIdfDF.join(tfIdfDFWithNames, Seq("term"))
                                .withColumn("mult", $"tf-idf_query" * $"tf-idf")
                                .groupBy("id", "name")
                                .agg(sum("mult") as "dot_product")
    
    // Join dot product
    val cosineSimilarityDF = dotProdDF.join(docSizeDF, Seq("id"))
                                      .withColumn("cosine_similarity", $"dot_product" / ($"size" * querySize))
                                      .orderBy(desc("cosine_similarity"))
  
    // Show only top ten movie names
    cosineSimilarityDF.select("name").show(10, false)
}}