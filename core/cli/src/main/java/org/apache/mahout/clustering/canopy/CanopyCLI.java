package org.apache.mahout.clustering.canopy;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;

public class CanopyCLI extends AbstractCLI {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new CanopyCLI(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.t1Option().create());
    addOption(DefaultOptionCreator.t2Option().create());
    addOption(DefaultOptionCreator.t3Option().create());
    addOption(DefaultOptionCreator.t4Option().create());
    addOption(DefaultOptionCreator.clusterFilterOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator.clusteringOption().create());
    addOption(DefaultOptionCreator.methodOption().create());
    addOption(DefaultOptionCreator.outlierThresholdOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    Configuration conf = getConf();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(conf, output);
    }
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    double t1 = Double.parseDouble(getOption(DefaultOptionCreator.T1_OPTION));
    double t2 = Double.parseDouble(getOption(DefaultOptionCreator.T2_OPTION));
    double t3 = t1;
    if (hasOption(DefaultOptionCreator.T3_OPTION)) {
      t3 = Double.parseDouble(getOption(DefaultOptionCreator.T3_OPTION));
    }
    double t4 = t2;
    if (hasOption(DefaultOptionCreator.T4_OPTION)) {
      t4 = Double.parseDouble(getOption(DefaultOptionCreator.T4_OPTION));
    }
    int clusterFilter = 0;
    if (hasOption(DefaultOptionCreator.CLUSTER_FILTER_OPTION)) {
      clusterFilter = Integer
          .parseInt(getOption(DefaultOptionCreator.CLUSTER_FILTER_OPTION));
    }
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    boolean runSequential = getOption(DefaultOptionCreator.METHOD_OPTION)
        .equalsIgnoreCase(DefaultOptionCreator.SEQUENTIAL_METHOD);
    DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);
    double clusterClassificationThreshold = 0.0;
    if (hasOption(DefaultOptionCreator.OUTLIER_THRESHOLD)) {
      clusterClassificationThreshold = Double.parseDouble(getOption(DefaultOptionCreator.OUTLIER_THRESHOLD));
    }
    CanopyDriver driver = new CanopyDriver();
    driver.run(conf, input, output, measure, t1, t2, t3, t4, clusterFilter,
        runClustering, clusterClassificationThreshold, runSequential );
    return 0;
  }

}
