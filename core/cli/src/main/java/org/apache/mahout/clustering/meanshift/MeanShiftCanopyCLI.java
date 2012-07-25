package org.apache.mahout.clustering.meanshift;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.kernel.IKernelProfile;

public class MeanShiftCanopyCLI extends AbstractCLI {
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new MeanShiftCanopyCLI(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator.inputIsCanopiesOption().create());
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.kernelProfileOption().create());
    addOption(DefaultOptionCreator.t1Option().create());
    addOption(DefaultOptionCreator.t2Option().create());
    addOption(DefaultOptionCreator.clusteringOption().create());
    addOption(DefaultOptionCreator.methodOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    String kernelProfileClass = getOption(DefaultOptionCreator.KERNEL_PROFILE_OPTION);
    double t1 = Double.parseDouble(getOption(DefaultOptionCreator.T1_OPTION));
    double t2 = Double.parseDouble(getOption(DefaultOptionCreator.T2_OPTION));
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    double convergenceDelta = Double
        .parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    int maxIterations = Integer
        .parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    boolean inputIsCanopies = hasOption(MeanShiftConfig.INPUT_IS_CANOPIES_OPTION);
    boolean runSequential = getOption(DefaultOptionCreator.METHOD_OPTION)
        .equalsIgnoreCase(DefaultOptionCreator.SEQUENTIAL_METHOD);
    DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);
    IKernelProfile kernelProfile = ClassUtils.instantiateAs(kernelProfileClass, IKernelProfile.class);
    MeanShiftCanopyDriver driver = new MeanShiftCanopyDriver();
    driver.run(getConf(), input, output, measure, kernelProfile, t1, t2,
        convergenceDelta, maxIterations, inputIsCanopies, runClustering,
        runSequential);
    return 0;
  }

}
