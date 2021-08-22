// TwitterStreaming.scala
//
//	Description:	Uses the Twitter API to obtain a constant stream of tweets
//						related to a provided topic. Then uses CoreNLP to perform
//						sentiment analysis on this stream of tweets to determine
//						whether the tweets regarding the topic are more positive,
//						negative, or neutral. The sentiments are then sent through
//						Kafka to be visualized by programs ElasticSearch, Kibana,
//						and LogStash so that a graph over time of the sentiment
//						can be recorded.
// 	Arguments:
//		<consumerKey> is the consumer key for the Twitter API
//		<consumerSecret> is the consumer secret for the Twitter API
//		<accessToken> is the access token for the Twitter API
//		<accessTokenSecret> is the access token secret for the Twitter API
//		<topic> is the topic being searched for
//		<kafkaTopic> is the name of the Kafka topic

import java.util.Properties

import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations.SentimentAnnotatedTree
import edu.stanford.nlp.util.CoreMap
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}
import org.apache.kafka.common.serialization.{StringDeserializer, StringSerializer}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.twitter._
import org.apache.spark.streaming.{Seconds, StreamingContext}

import scala.collection.JavaConverters.asScalaBufferConverter

object TwitterStreaming {

  def main(args: Array[String]): Unit = {

    // Confirm correct arguments
    if (args.length != 6) {
      println("Correct usage: TwitterStreaming <twitter-consumer-key> <twitter-consumer-secret> <twitter-access-token> <twitter-access-token-secret> <topic> <kafka-topic>")
      System.exit(1)
    }

    // Get arguments
    val consumerKey = args(0)
    val consumerSecret = args(1)
    val accessToken = args(2)
    val accessTokenSecret = args(3)
    val topic = args(4)
    val kafkaTopic = args(5)

    // Configuration for Twitter API
    System.setProperty("twitter4j.oauth.consumerKey", consumerKey)
    System.setProperty("twitter4j.oauth.consumerSecret", consumerSecret)
    System.setProperty("twitter4j.oauth.accessToken", accessToken)
    System.setProperty("twitter4j.oauth.accessTokenSecret", accessTokenSecret)

    // Initialize spark
    val spark = SparkSession.builder()
      .appName("twitter-streaming")
      .master("local[*]")
      .getOrCreate()

    // Create broadcast variable for CoreNLP pipeline properties
    val props = new Properties()
    props.put("annotators", "tokenize, ssplit, parse, sentiment")
    val propsBC = spark.sparkContext.broadcast(props)

    // Function to map sentiment integer (returned from CoreNLP pipeline) to string
    def getSentiment(sentiment: Int): String = sentiment match {
      case x if x == 0 || x == 1 => "NEGATIVE"
      case 2 => "NEUTRAL"
      case x if x == 3 || x == 4 => "POSITIVE"
    }

    // Perform sentiment analysis using CoreNLP
    def extractSentiments(text: String): List[String] = {
      val pipeline = new StanfordCoreNLP(propsBC.value)
      val doc = new Annotation(text)
      pipeline.annotate(doc)
      val sentences: List[CoreMap] = doc.get(classOf[CoreAnnotations.SentencesAnnotation]).asScala.toList
      sentences
        .map(sentence => sentence.get(classOf[SentimentAnnotatedTree]))
        .map(tree => getSentiment(RNNCoreAnnotations.getPredictedClass(tree)))
    }

    // Initialize logger
    val rootLogger = Logger.getRootLogger
    rootLogger.setLevel(Level.ERROR)

    // Set Kafka producer properties
    val producerProps =  new java.util.Properties()
    producerProps.put("bootstrap.servers", "localhost:9092")
    producerProps.put("key.serializer", classOf[StringSerializer])
    producerProps.put("value.serializer", classOf[StringSerializer])
    producerProps.put("key.deserializer", classOf[StringDeserializer])
    producerProps.put("value.deserializer", classOf[StringDeserializer])

    // Initialize Kafka producer
    val producer = new KafkaProducer[String, String](producerProps)

    // Initialize stream for tweets related to topic
    val ssc = new StreamingContext(spark.sparkContext, Seconds(10))
    val stream = TwitterUtils.createStream(ssc, None, filters = Seq(topic))

    // Perform sentiment analysis on tweets
    val sentiments = stream
      .map(_.getText)
      .flatMap(extractSentiments)

    // Print sentiments to console
    sentiments.print()

    // Send sentiments to kafka
    sentiments.foreachRDD(rdd => {
      rdd.collect().foreach { value =>
        producer.send( new ProducerRecord(kafkaTopic, value) )
      }
    })

    // Start stream and wait for termination
    ssc.start()
    ssc.awaitTermination()
  }
}
