package org.apache.mahout.classifier.naivebayes.training;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

public class TrainNaiveBayesCLI extends AbstractCLI {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new TrainNaiveBayesCLI(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("labels", "l", "comma-separated list of labels to include in training", false);

    addOption(buildOption("extractLabels", "el", "Extract the labels from the input", false, false, ""));
    addOption("alphaI", "a", "smoothing parameter", String.valueOf(1.0f));
    addOption(buildOption("trainComplementary", "c", "train complementary?", false, false, String.valueOf(false)));
    addOption("labelIndex", "li", "The path to store the label index in", false);
    addOption(DefaultOptionCreator.overwriteOption().create());
    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), getOutputPath());
      HadoopUtil.delete(getConf(), getTempPath());
    }
    Path labPath;
    String labPathStr = getOption("labelIndex");
    if (labPathStr != null) {
      labPath = new Path(labPathStr);
    } else {
      labPath = getTempPath("labelIndex");
    }
    float alphaI = Float.parseFloat(getOption("alphaI"));
    boolean trainComplementary = Boolean.parseBoolean(getOption("trainComplementary"));
    TrainNaiveBayesJob job = new TrainNaiveBayesJob();
    String labels = null;
    if (hasOption("labels")) {
      labels = getOption("labels");
    }
    return job.run(getInputPath(), getOutputPath(), labPath, alphaI, trainComplementary, labels);
  }
}
