package org.apache.mahout.math.hadoop.decomposer;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

public class EigenVerificationCLI extends AbstractCLI {

  @Override
  public int run(String[] args) throws Exception {
    Map<String, List<String>> argMap = handleArgs(args);
    if (argMap == null) {
      return -1;
    }
    if (argMap.isEmpty()) {
      return 0;
    }
    // parse out the arguments
    runJob(getConf(),
           new Path(getOption("eigenInput")),
           new Path(getOption("corpusInput")),
           getOutputPath(),
           getOption("inMemory") != null,
           Double.parseDouble(getOption("maxError")),
           //Double.parseDouble(getOption("minEigenvalue")),
           Integer.parseInt(getOption("maxEigens")));
    return 0;
  }

  private Map<String, List<String>> handleArgs(String[] args) throws IOException {
    addOutputOption();
    addOption("eigenInput",
              "ei",
              "The Path for purported eigenVector input files (SequenceFile<WritableComparable,VectorWritable>.",
              null);
    addOption("corpusInput", "ci", "The Path for corpus input files (SequenceFile<WritableComparable,VectorWritable>.");
    addOption(DefaultOptionCreator.outputOption().create());
    addOption(DefaultOptionCreator.helpOption());
    addOption("inMemory", "mem", "Buffer eigen matrix into memory (if you have enough!)", "false");
    addOption("maxError", "err", "Maximum acceptable error", "0.05");
    addOption("minEigenvalue", "mev", "Minimum eigenvalue to keep the vector for", "0.0");
    addOption("maxEigens", "max", "Maximum number of eigenvectors to keep (0 means all)", "0");

    return parseArguments(args);
  }

  public void runJob(Configuration conf,
      Path eigenInput,
      Path corpusInput,
      Path output,
      boolean inMemory,
      double maxError,
      int maxEigens) throws IOException {
    new EigenVerificationJob().runJob(conf, eigenInput, corpusInput, output, inMemory, maxError, maxEigens);
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new EigenVerificationCLI(), args);
  }

}
