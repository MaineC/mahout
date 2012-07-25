/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.clustering.fuzzykmeans;

import static org.apache.mahout.clustering.topdown.PathDirectory.CLUSTERED_POINTS_DIRECTORY;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassificationDriver;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.iterator.ClusterIterator;
import org.apache.mahout.clustering.iterator.ClusteringPolicy;
import org.apache.mahout.clustering.iterator.FuzzyKMeansClusteringPolicy;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FuzzyKMeansDriver extends Configured {

  public static final String M_OPTION = "m";

  private static final Logger log = LoggerFactory.getLogger(FuzzyKMeansDriver.class);


  /**
   * Iterate over the input vectors to produce clusters and, if requested, use the
   * results of the final iteration to cluster the input vectors.
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param m
   *          the fuzzification factor, see
   *          http://en.wikipedia.org/wiki/Data_clustering#Fuzzy_c-means_clustering
   * @param runClustering 
   *          true if points are to be clustered after iterations complete
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold 
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   * @param runSequential if true run in sequential execution mode
   */
  public static void run(Path input,
                         Path clustersIn,
                         Path output,
                         DistanceMeasure measure,
                         double convergenceDelta,
                         int maxIterations,
                         float m,
                         boolean runClustering,
                         boolean emitMostLikely,
                         double threshold,
                         boolean runSequential) throws IOException, ClassNotFoundException, InterruptedException {
    Path clustersOut = buildClusters(new Configuration(),
                                     input,
                                     clustersIn,
                                     output,
                                     measure,
                                     convergenceDelta,
                                     maxIterations,
                                     m,
                                     runSequential);
    if (runClustering) {
      log.info("Clustering ");
      clusterData(input,
                  clustersOut,
                  output,
                  measure,
                  convergenceDelta,
                  m,
                  emitMostLikely,
                  threshold,
                  runSequential);
    }
  }

  /**
   * Iterate over the input vectors to produce clusters and, if requested, use the
   * results of the final iteration to cluster the input vectors.
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param m
   *          the fuzzification factor, see
   *          http://en.wikipedia.org/wiki/Data_clustering#Fuzzy_c-means_clustering
   * @param runClustering 
   *          true if points are to be clustered after iterations complete
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold 
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   * @param runSequential if true run in sequential execution mode
   */
  public static void run(Configuration conf,
                         Path input,
                         Path clustersIn,
                         Path output,
                         DistanceMeasure measure,
                         double convergenceDelta,
                         int maxIterations,
                         float m,
                         boolean runClustering,
                         boolean emitMostLikely,
                         double threshold,
                         boolean runSequential)
    throws IOException, ClassNotFoundException, InterruptedException {
    Path clustersOut =
        buildClusters(conf, input, clustersIn, output, measure, convergenceDelta, maxIterations, m, runSequential);
    if (runClustering) {
      log.info("Clustering");
      clusterData(input,
                  clustersOut,
                  output,
                  measure,
                  convergenceDelta,
                  m,
                  emitMostLikely,
                  threshold,
                  runSequential);
    }
  }

  /**
   * Iterate over the input vectors to produce cluster directories for each iteration
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the file pathname for initial cluster centers
   * @param output
   *          the directory pathname for output points
   * @param measure
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param m
   *          the fuzzification factor, see
   *          http://en.wikipedia.org/wiki/Data_clustering#Fuzzy_c-means_clustering
   * @param runSequential if true run in sequential execution mode
   * 
   * @return the Path of the final clusters directory
   */
  public static Path buildClusters(Configuration conf,
                                   Path input,
                                   Path clustersIn,
                                   Path output,
                                   DistanceMeasure measure,
                                   double convergenceDelta,
                                   int maxIterations,
                                   float m,
                                   boolean runSequential)
    throws IOException, InterruptedException, ClassNotFoundException {
    
    List<Cluster> clusters = new ArrayList<Cluster>();
    FuzzyKMeansUtil.configureWithClusterInfo(clustersIn, clusters);
    
    if (conf==null) {
      conf = new Configuration();
    }
    
    if (clusters.isEmpty()) {
      throw new IllegalStateException("No input clusters found. Check your -c argument.");
    }
    
    Path priorClustersPath = new Path(output, Cluster.INITIAL_CLUSTERS_DIR);   
    ClusteringPolicy policy = new FuzzyKMeansClusteringPolicy(m, convergenceDelta);
    ClusterClassifier prior = new ClusterClassifier(clusters, policy);
    prior.writeToSeqFiles(priorClustersPath);
    
    if (runSequential) {
      new ClusterIterator().iterateSeq(conf, input, priorClustersPath, output, maxIterations);
    } else {
      new ClusterIterator().iterateMR(conf, input, priorClustersPath, output, maxIterations);
    }
    return output;
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for input clusters
   * @param output
   *          the directory pathname for output points
   * @param measure
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   * @param runSequential if true run in sequential execution mode
   */
  public static void clusterData(Path input,
                                 Path clustersIn,
                                 Path output,
                                 DistanceMeasure measure,
                                 double convergenceDelta,
                                 float m,
                                 boolean emitMostLikely,
                                 double threshold,
                                 boolean runSequential)
    throws IOException, ClassNotFoundException, InterruptedException {
    
    ClusterClassifier.writePolicy(new FuzzyKMeansClusteringPolicy(m, convergenceDelta), clustersIn);
    ClusterClassificationDriver.run(input, output, new Path(output, CLUSTERED_POINTS_DIRECTORY), threshold, true,
        runSequential);
  }
}
