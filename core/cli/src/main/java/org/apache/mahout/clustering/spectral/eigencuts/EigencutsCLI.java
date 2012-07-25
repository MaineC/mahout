package org.apache.mahout.clustering.spectral.eigencuts;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

public class EigencutsCLI extends AbstractCLI {
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new EigencutsCLI(), args);
  }

  @Override
  public int run(String[] arg0) throws Exception {

    // set up command line arguments
    addOption("half-life", "b", "Minimal half-life threshold", true);
    addOption("dimensions", "d", "Square dimensions of affinity matrix", true);
    addOption("epsilon", "e", "Half-life threshold coefficient", Double.toString(EigencutsDriver.EPSILON_DEFAULT));
    addOption("tau", "t", "Threshold for cutting affinities", Double.toString(EigencutsDriver.TAU_DEFAULT));
    addOption("eigenrank", "k", "Number of top eigenvectors to use", true);
    addOption(DefaultOptionCreator.inputOption().create());
    addOption(DefaultOptionCreator.outputOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    Map<String, List<String>> parsedArgs = parseArguments(arg0);
    if (parsedArgs == null) {
      return 0;
    }

    // read in the command line values
    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    int dimensions = Integer.parseInt(getOption("dimensions"));
    double halflife = Double.parseDouble(getOption("half-life"));
    double epsilon = Double.parseDouble(getOption("epsilon"));
    double tau = Double.parseDouble(getOption("tau"));
    int eigenrank = Integer.parseInt(getOption("eigenrank"));

    EigencutsDriver driver = new EigencutsDriver();
    driver.run(getConf(), input, output, eigenrank, dimensions, halflife, epsilon, tau);
    return 0;
  }
}
