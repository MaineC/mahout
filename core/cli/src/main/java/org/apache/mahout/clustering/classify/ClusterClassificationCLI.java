package org.apache.mahout.clustering.classify;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

public class ClusterClassificationCLI extends AbstractCLI {
  /**
   * CLI to run Cluster Classification Driver.
   */
  @Override
  public int run(String[] args) throws Exception {
    
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.methodOption().create());
    addOption(DefaultOptionCreator.clustersInOption()
        .withDescription("The input centroids, as Vectors.  Must be a SequenceFile of Writable, Cluster/Canopy.")
        .create());
    
    if (parseArguments(args) == null) {
      return -1;
    }
    
    Path input = getInputPath();
    Path output = getOutputPath();
    
    if (getConf() == null) {
      setConf(new Configuration());
    }
    Path clustersIn = new Path(getOption(DefaultOptionCreator.CLUSTERS_IN_OPTION));
    boolean runSequential = getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(
        DefaultOptionCreator.SEQUENTIAL_METHOD);
    
    double clusterClassificationThreshold = 0.0;
    if (hasOption(DefaultOptionCreator.OUTLIER_THRESHOLD)) {
      clusterClassificationThreshold = Double.parseDouble(getOption(DefaultOptionCreator.OUTLIER_THRESHOLD));
    }
    
    ClusterClassificationDriver driver = new ClusterClassificationDriver();
    driver.run(input, clustersIn, output, clusterClassificationThreshold, true, runSequential);
    
    return 0;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new ClusterClassificationCLI(), args);
  }
  
  

}
