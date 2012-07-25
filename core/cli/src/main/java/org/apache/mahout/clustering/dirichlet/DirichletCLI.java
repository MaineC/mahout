package org.apache.mahout.clustering.dirichlet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.dirichlet.models.DistributionDescription;
import org.apache.mahout.clustering.dirichlet.models.GaussianClusterDistribution;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.RandomAccessSparseVector;

public class DirichletCLI extends AbstractCLI {
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new DirichletCLI(), args);
  }
  
  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.numClustersOption().withRequired(true).create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator.clusteringOption().create());
    addOption(DirichletDriver.ALPHA_OPTION, "a0", "The alpha0 value for the DirichletDistribution. Defaults to 1.0", "1.0");
    addOption(DirichletDriver.MODEL_DISTRIBUTION_CLASS_OPTION, "md",
        "The ModelDistribution class name. Defaults to GaussianClusterDistribution",
        GaussianClusterDistribution.class.getName());
    addOption(DirichletDriver.MODEL_PROTOTYPE_CLASS_OPTION, "mp",
        "The ModelDistribution prototype Vector class name. Defaults to RandomAccessSparseVector",
        RandomAccessSparseVector.class.getName());
    addOption(DefaultOptionCreator.distanceMeasureOption().withRequired(false).create());
    addOption(DefaultOptionCreator.emitMostLikelyOption().create());
    addOption(DefaultOptionCreator.thresholdOption().create());
    addOption(DefaultOptionCreator.methodOption().create());
    
    if (parseArguments(args) == null) {
      return -1;
    }
    
    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    String modelFactory = getOption(DirichletDriver.MODEL_DISTRIBUTION_CLASS_OPTION);
    String modelPrototype = getOption(DirichletDriver.MODEL_PROTOTYPE_CLASS_OPTION);
    String distanceMeasure = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    int numModels = Integer.parseInt(getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    boolean emitMostLikely = Boolean.parseBoolean(getOption(DefaultOptionCreator.EMIT_MOST_LIKELY_OPTION));
    double threshold = Double.parseDouble(getOption(DefaultOptionCreator.THRESHOLD_OPTION));
    double alpha0 = Double.parseDouble(getOption(DirichletDriver.ALPHA_OPTION));
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    boolean runSequential = getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(
        DefaultOptionCreator.SEQUENTIAL_METHOD);
    int prototypeSize = DirichletDriver.readPrototypeSize(input);
    
    DistributionDescription description = new DistributionDescription(modelFactory, modelPrototype, distanceMeasure,
        prototypeSize);

    DirichletDriver driver = new DirichletDriver();
    driver.run(getConf(), input, output, description, numModels, maxIterations, alpha0, runClustering, emitMostLikely,
        threshold, runSequential);
    return 0;
  }

}
