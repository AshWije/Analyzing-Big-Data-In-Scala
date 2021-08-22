// PageRank.scala
//
//	Description:	Contains a scala class called PageRank that will run the page rank
//						algorithm on the input airport data for a given number of
//						iterations. This is done in order to measure the importance
//						(rank) of each airport.
// 	Inputs:
//		<inputLoc> is the location of the airport data csv file
//		<maxIterations> is an integer representing the maximum iterations for page rank

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, collect_set, explode, lit}

class PageRank(val inputLoc: String, val maxIterations: Int) {
  val alpha = 0.15
  val initPRVal = 10.0

  // Create spark session
  val spark : SparkSession = SparkSession.builder
    .appName("Page Rank")
    .getOrCreate()

  def getPageRank: RDD[(String, Double)] = {

    // Read airport data from input location argument
    val airportData = spark.read.option("header", "true").csv(inputLoc).select("ORIGIN_AIRPORT_ID", "ORIGIN", "DEST_AIRPORT_ID", "DEST")

    // Get out links (number of flights LEAVING airport)
    val outLinkDF = airportData.groupBy("ORIGIN_AIRPORT_ID", "ORIGIN").count().toDF("id", "code", "outLink")

    // Get in links (list of flights ARRIVING at airport)
    val inLinkDF = airportData.groupBy("DEST_AIRPORT_ID", "DEST").agg(collect_set(col("ORIGIN_AIRPORT_ID"))).toDF("id", "code", "inLinkList")

    // Do a full outer join
    val inAndOutLinkDF = outLinkDF.join(inLinkDF, Seq("id", "code"), "full")

    // Get the total number of airports
    val N = inAndOutLinkDF.count().toDouble

    // Add page rank column with value initPRVal for all airports
    var PageRankDF = inAndOutLinkDF.withColumn("pageRank", lit(initPRVal))

    // Loop through iterations
    for(_ <- 1 to maxIterations) {

      // Explode in link list
      val explodedInLinkList = PageRankDF.withColumn("inLink_id", explode(col("inLinkList")))

      // Rename page rank columns (to simplify join)
      val renamedPR = PageRankDF.toDF("inLink_id", "inLink_code", "inLink_outLink", "inLink_list", "inLink_pageRank")

      // Join explodedInLinkList with renamedPR
      val joined = explodedInLinkList.join(renamedPR, Seq("inLink_id"))

      // Calculate PR(t) / C(t)
      val withPR_C = joined.withColumn("inLink_PR_C", col("inLink_pageRank") / col("inLink_outLink"))

      // Calculate sum of PR(t) / C(t) for all airports t that are in links
      val withSum = withPR_C.groupBy("id").sum("inLink_PR_C").toDF("id", "sum")

      // Join withSum with PageRankDF
      val PRBeforeUpdate = withSum.join(PageRankDF, Seq("id"))

      // Update page ranks
      PageRankDF = PRBeforeUpdate.withColumn("pageRank", lit(alpha / N) + lit(1 - alpha) * col("sum")).drop("sum")
    }

    // Get only airport code and page rank columns
    PageRankDF = PageRankDF.select("code", "pageRank")

    // Convert DF to RDD
    val PageRankRDD = PageRankDF.rdd.map(row => (row(0).toString, row(1).asInstanceOf[Double]))

    // Return RDD sorted by descending page rank
    PageRankRDD.sortBy(-_._2)
  }
}
