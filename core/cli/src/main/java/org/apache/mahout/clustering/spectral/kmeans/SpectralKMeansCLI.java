package org.apache.mahout.clustering.spectral.kmeans;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;

public class SpectralKMeansCLI extends AbstractCLI {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SpectralKMeansCLI(), args);
  }

  @Override
  public int run(String[] arg0)
    throws IOException, ClassNotFoundException, InstantiationException, IllegalAccessException, InterruptedException {
    // set up command line options
    Configuration conf = getConf();
    addInputOption();
    addOutputOption();
    addOption("dimensions", "d", "Square dimensions of affinity matrix", true);
    addOption("clusters", "k", "Number of clusters and top eigenvectors", true);
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    Map<String, List<String>> parsedArgs = parseArguments(arg0);
    if (parsedArgs == null) {
      return 0;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(conf, output);
    }
    int numDims = Integer.parseInt(getOption("dimensions"));
    int clusters = Integer.parseInt(getOption("clusters"));
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);
    double convergenceDelta = Double.parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    
    SpectralKMeansDriver driver = new SpectralKMeansDriver();
    driver.run(conf, input, output, numDims, clusters, measure, convergenceDelta, maxIterations);

    return 0;
  }

}
