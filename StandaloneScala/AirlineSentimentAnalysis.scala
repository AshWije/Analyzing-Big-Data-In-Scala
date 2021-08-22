// AirlineSentimentAnalysis.scala
//
//	Description:	Contains a scala class called AirlineSentimentAnalysis that will create a model
//						used to perform sentiment analysis on the input airline tweets. This is
//						done in order to determine whether a particular airline has more positive
//						or negative reviews.
// 	Inputs:
//		<inputLoc> is the location of the tweets csv file

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.{SparkConf, SparkContext}

class AirlineSentimentAnalysis(val inputLoc: String) {

  // Create Spark context with Spark configuration
  val sc = new SparkContext(new SparkConf().setAppName("AirlineSentimentAnalysis"))

  // Create spark session
  val spark : SparkSession = SparkSession.builder
    .appName("AirlineSentimentAnalysis")
    .getOrCreate()

  def runPipeline: RDD[String] = {

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LOADING:
    // Read tweet data from input location argument and remove rows where text field is null
    val tweetData = spark.read.option("header", "true").csv(inputLoc)
      .select("text", "airline_sentiment")
      .filter(col("text") =!= "null")

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PRE-PROCESSING:
    // Tokenizer: Transform the text column into words by breaking down the sentence into words
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    // Stop word remover: Remove stop-words from the text column
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    // Term Hashing: Convert words to term-frequency vectors
    val hashingTF = new HashingTF()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("features")

    // Label Conversion: The label is a string e.g. “Positive”, which you need to convert to numeric format
    val indexer = new StringIndexer()
      .setInputCol("airline_sentiment")
      .setOutputCol("label")

    // Logistic Regression
    val lr = new LogisticRegression()

    // Define pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, indexer, lr))

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MODEL CREATION
    // ParameterGridBuilder for parameter tuning
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.maxIter, Array(5, 10, 20))
      .addGrid(lr.elasticNetParam, Array(0.1, 0.01))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .build()

    // CrossValidator for finding the best model parameters
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)
      .setParallelism(2)

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MODEL TESTING AND CROSS VALIDATION
    // Run cross-validation and choose the best set of parameters
    val cvModel = cv.fit(tweetData)

    // Get predictions for tweet data
    val predictionsDF = cvModel.transform(tweetData)

    // Convert DF to RDD with form (prediction, label)
    val predictionAndLabelRDD = predictionsDF.select("prediction", "label").rdd
      .map(row => (row(0).asInstanceOf[Double], row(1).asInstanceOf[Double]))

    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabelRDD)

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // CREATE OUTPUT
    // Confusion matrix
    var out = "Confusion matrix:\n" + metrics.confusionMatrix

    // Overall Statistics
    val accuracy = metrics.accuracy
    out += s"\n\nAccuracy = $accuracy"

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      out += s"\nPrecision($l) = " + metrics.precision(l)
    }

    // Recall by label
    labels.foreach { l =>
      out += s"\nRecall($l) = " + metrics.recall(l)
    }

    // False positive rate by label
    labels.foreach { l =>
      out += s"\nFPR($l) = " + metrics.falsePositiveRate(l)
    }

    // F-measure by label
    labels.foreach { l =>
      out += s"\nF1-Score($l) = " + metrics.fMeasure(l)
    }

    // Weighted stats
    out += s"\nWeighted precision: ${metrics.weightedPrecision}"
    out += s"\nWeighted recall: ${metrics.weightedRecall}"
    out += s"\nWeighted F1 score: ${metrics.weightedFMeasure}"
    out += s"\nWeighted false positive rate: ${metrics.weightedFalsePositiveRate}"

    // Convert output string into RDD (to simplify saving)
    sc.parallelize(".").map(_ => out)
  }
}