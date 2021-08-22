// EpinionsGraphFrameAnalysis.scala
//
//	Description:	Creates a GraphFrame data structure to represent the social network
//						Epinions that is provided as input. Epinions is a consumer review
//						website where people can choose to trust one another. The GraphFrame
//						nodes represent the users on Epinions, while graph edges represent
//						the trust from one user to another user. This GraphFrame is used to
//						analyze and output several aspects of Epinions:
//							* Top five most trusting users 		(highest out degree)
//							* Top five most trustworthy users 	(highest in degree)
//							* Top five most important users 	(highest page rank)
//							* Top five communities 				(highest connected components)
//							* Top five trust networks 			(highest triangle count)
//						This was used with AWS EMR.
// 	Arguments:
//		<inputLoc> is the location of the input file (from: https://snap.stanford.edu/data/soc-Epinions1.html)
//		<outputLoc> is the name of the directory where the output will be placed
//		<checkPointLoc> is the location of spark checkpoint directory

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.desc
import org.graphframes.GraphFrame

object EpinionsGraphFrameAnalysis {
  def main(args: Array[String]): Unit = {

    // Initialize spark
    val spark = SparkSession.builder()
      .appName("social-network-analysis")
      .getOrCreate()
    import spark.implicits._

    // Confirm correct arguments
    if (args.length != 3) {
      println("Correct usage: EpinionsGraphFrameAnalysis <path-to-input> <path-to-output> <spark-checkpoint-directory>")
      System.exit(1)
    }

    // Get arguments
    val inputLoc = args(0)
    val outputLoc = args(1)
    val checkPointLoc = args(2)

    // Set spark checkpoint directory
    spark.sparkContext.setCheckpointDir(checkPointLoc)

    // LOAD DATA
    // Link to dataset -> https://snap.stanford.edu/data/soc-Epinions1.html
    val data = spark.read
      .option("comment", "#")
      .option("delimiter", "\t")
      .csv(inputLoc)
      .toDF("FromNodeId", "ToNodeId")

    // CREATE GRAPH
    // Define the vertices and edges
    val vertices = (0 to 75879).toDF("id")
    val edges = data.toDF("src", "dst")

    // Define the graph and cache it
    val graph = GraphFrame(vertices, edges)
    graph.cache()

	// TOP FIVE MOST TRUSTING USERS
    // Find the top 5 nodes with the highest outdegree and find the count of the number of outgoing edges in each
    val top5OutDegree = graph.outDegrees
      .orderBy(desc("outDegree"))
      .limit(5)

    // Write output
    top5OutDegree.write
      .mode("overwrite")
      .format("csv")
      .option("header", "true")
      .save(outputLoc + "/top5OutDegree_out")

    // TOP FIVE MOST TRUSTWORTHY USERS
    // Find the top 5 nodes with the highest indegree and find the count of the number of incoming edges in each
    val top5InDegree = graph.inDegrees
      .orderBy(desc("inDegree"))
      .limit(5)

    // Write output
    top5InDegree.write
      .mode("overwrite")
      .format("csv")
      .option("header", "true")
      .save(outputLoc + "/top5InDegree_out")

    // TOP FIVE MOST IMPORTANT USERS
    // Calculate PageRank for each of the nodes and output the top 5 nodes with the highest PageRank values
    // Run PageRank until convergence to tolerance "tol"
    val pageRankGF = graph.pageRank.resetProbability(0.15).tol(0.01).run()
    val top5PageRank = pageRankGF.vertices.select("id", "pagerank")
      .orderBy(desc("pagerank"))
      .limit(5)

    // Write output
    top5PageRank.write
      .mode("overwrite")
      .format("csv")
      .option("header", "true")
      .save(outputLoc + "/top5PageRank_out")

    // TOP FIVE COMMUNITIES
    // Run the connected components algorithm on it and find the top 5 components with the largest number of nodes
    val connectedComponentsDF = graph.connectedComponents.run()
    val top5Components = connectedComponentsDF.groupBy("component").count()
      .orderBy(desc("count"))
      .limit(5)

    // Write output
    top5Components.write
      .mode("overwrite")
      .format("csv")
      .option("header", "true")
      .save(outputLoc + "/top5Components_out")

    // TOP FIVE TRUST NETWORKS
    // Run the triangle counts algorithm on each of the vertices and output the top 5 vertices with the largest triangle count
    val triangleCountDF = graph.triangleCount.run()
    val top5TriangleCounts = triangleCountDF.select("id", "count")
      .orderBy(desc("count"))
      .limit(5)

    // Write output
    top5TriangleCounts.write
      .mode("overwrite")
      .format("csv")
      .option("header", "true")
      .save(outputLoc + "/top5TriangleCounts_out")
  }
}
