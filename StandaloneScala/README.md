# Standalone Scala

This folder contains four standalone scala programs that perform the following tasks aimed at analyzing
big data:


**Airline Sentiment Analysis**
Contains a scala class called AirlineSentimentAnalysis that will create a model
used to perform sentiment analysis on the input airline tweets. This is
done in order to determine whether a particular airline has more positive
or negative reviews.


**Epinions GraphFrame Analysis**
Creates a GraphFrame data structure to represent the social network
Epinions that is provided as input. Epinions is a consumer review
website where people can choose to trust one another. The GraphFrame
nodes represent the users on Epinions, while graph edges represent
the trust from one user to another user. This GraphFrame is used to
analyze and output several aspects of Epinions:
* Top five most trusting users 		(highest out degree)
* Top five most trustworthy users 	(highest in degree)
* Top five most important users 	(highest page rank)
* Top five communities 				(highest connected components)
* Top five trust networks 			(highest triangle count)

This was used with AWS EMR.


**Page Rank**
Contains a scala class called PageRank that will run the page rank
algorithm on the input airport data for a given number of
iterations. This is done in order to measure the importance
(rank) of each airport.


**Twitter Streaming**
Uses the Twitter API to obtain a constant stream of tweets
related to a provided topic. Then uses CoreNLP to perform
sentiment analysis on this stream of tweets to determine
whether the tweets regarding the topic are more positive,
negative, or neutral. The sentiments are then sent through
Kafka to be visualized by programs ElasticSearch, Kibana,
and LogStash so that a graph over time of the sentiment
can be recorded.


2020