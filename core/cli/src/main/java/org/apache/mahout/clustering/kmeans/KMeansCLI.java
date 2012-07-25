package org.apache.mahout.clustering.kmeans;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;

public class KMeansCLI extends AbstractCLI {
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new KMeansCLI(), args);
  }
  
  @Override
  public int run(String[] args) throws Exception {
    
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator
        .clustersInOption()
        .withDescription(
            "The input centroids, as Vectors.  Must be a SequenceFile of Writable, Cluster/Canopy.  "
                + "If k is also specified, then a random set of vectors will be selected"
                + " and written out to this path first").create());
    addOption(DefaultOptionCreator
        .numClustersOption()
        .withDescription(
            "The k in k-Means.  If specified, then a random selection of k Vectors will be chosen"
                + " as the Centroid and written to the clusters input path.").create());
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator.clusteringOption().create());
    addOption(DefaultOptionCreator.methodOption().create());
    addOption(DefaultOptionCreator.outlierThresholdOption().create());
    
    if (parseArguments(args) == null) {
      return -1;
    }
    
    Path input = getInputPath();
    Path clusters = new Path(getOption(DefaultOptionCreator.CLUSTERS_IN_OPTION));
    Path output = getOutputPath();
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    if (measureClass == null) {
      measureClass = SquaredEuclideanDistanceMeasure.class.getName();
    }
    double convergenceDelta = Double.parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);
    
    if (hasOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)) {
      clusters = RandomSeedGenerator.buildRandom(getConf(), input, clusters,
          Integer.parseInt(getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)), measure);
    }
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    boolean runSequential = getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(
        DefaultOptionCreator.SEQUENTIAL_METHOD);
    if (getConf() == null) {
      setConf(new Configuration());
    }
    double clusterClassificationThreshold = 0.0;
    if (hasOption(DefaultOptionCreator.OUTLIER_THRESHOLD)) {
      clusterClassificationThreshold = Double.parseDouble(getOption(DefaultOptionCreator.OUTLIER_THRESHOLD));
    }
    KMeansDriver driver = new KMeansDriver();
    driver.run(getConf(), input, clusters, output, measure, convergenceDelta, maxIterations, runClustering,
        clusterClassificationThreshold, runSequential);
    return 0;
  }
}
