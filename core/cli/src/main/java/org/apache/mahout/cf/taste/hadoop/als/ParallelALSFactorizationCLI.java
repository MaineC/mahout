package org.apache.mahout.cf.taste.hadoop.als;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;

public class ParallelALSFactorizationCLI extends AbstractCLI {


  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ParallelALSFactorizationCLI(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("lambda", null, "regularization parameter", true);
    addOption("implicitFeedback", null, "data consists of implicit feedback?", String.valueOf(false));
    addOption("alpha", null, "confidence parameter (only used on implicit feedback)", String.valueOf(40));
    addOption("numFeatures", null, "dimension of the feature space", true);
    addOption("numIterations", null, "number of iterations", true);

    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    int numFeatures = Integer.parseInt(getOption("numFeatures"));
    int numIterations = Integer.parseInt(getOption("numIterations"));
    double lambda = Double.parseDouble(getOption("lambda"));
    double alpha = Double.parseDouble(getOption("alpha"));
    boolean implicitFeedback = Boolean.parseBoolean(getOption("implicitFeedback"));

    ParallelALSFactorizationJob job = new ParallelALSFactorizationJob();
    return job.run(getInputPath(), getOutputPath(),
        numFeatures, numIterations, lambda, alpha, implicitFeedback);
  }
}
