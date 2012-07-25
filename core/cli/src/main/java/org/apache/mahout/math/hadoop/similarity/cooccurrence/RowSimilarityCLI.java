package org.apache.mahout.math.hadoop.similarity.cooccurrence;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasures;

public class RowSimilarityCLI extends AbstractCLI {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RowSimilarityCLI(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("numberOfColumns", "r", "Number of columns in the input matrix", false);
    addOption("similarityClassname", "s", "Name of distributed similarity class to instantiate, alternatively use "
        + "one of the predefined similarities (" + VectorSimilarityMeasures.list() + ')');
    addOption("maxSimilaritiesPerRow", "m", "Number of maximum similarities per row (default: "
        + RowSimilarityConfig.DEFAULT_MAX_SIMILARITIES_PER_ROW + ')', String.valueOf(RowSimilarityConfig.DEFAULT_MAX_SIMILARITIES_PER_ROW));
    addOption("excludeSelfSimilarity", "ess", "compute similarity of rows to themselves?", String.valueOf(false));
    addOption("threshold", "tr", "discard row pairs with a similarity value below this", false);
    addOption(DefaultOptionCreator.overwriteOption().create());

    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }
    RowSimilarityConfig conf = new RowSimilarityConfig();
    
    if (hasOption("numberOfColumns")) {
      // Number of columns explicitly specified via CLI
      conf.setNumberOfCols(Integer.parseInt(getOption("numberOfColumns")));
    } else {
      // else get the number of columns by determining the cardinality of a vector in the input matrix
      conf.setNumberOfCols(getDimensions(getInputPath()));
    }

    String similarityClassnameArg = getOption("similarityClassname");
    try {
      conf.setSimilarityClassName(VectorSimilarityMeasures.valueOf(similarityClassnameArg).getClassname());
    } catch (IllegalArgumentException iae) {
      conf.setSimilarityClassName(similarityClassnameArg);
    }

    // Clear the output and temp paths if the overwrite option has been set
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      // Clear the temp path
      HadoopUtil.delete(getConf(), getTempPath());
      // Clear the output path
      HadoopUtil.delete(getConf(), getOutputPath());
    }

    conf.setMaxSimilaritiesPerRow(Integer.parseInt(getOption("maxSimilaritiesPerRow")));
    conf.setExcludeSelfSimilarity(Boolean.parseBoolean(getOption("excludeSelfSimilarity")));
    if (hasOption("threshold")) {
        conf.setThreshold(Double.parseDouble(getOption("threshold")));
    } else {
      conf.setThreshold(RowSimilarityJob.NO_THRESHOLD);
    }


    String startPhase = AbstractCLI.getOption(parsedArgs, "--startPhase");
    Integer start = startPhase == null ? null : new Integer(startPhase);
    String endPhase = AbstractCLI.getOption(parsedArgs, "--endPhase");
    Integer end = endPhase == null ? null : new Integer(endPhase);
    RowSimilarityJob job = new RowSimilarityJob();
    return job.run(conf, getInputPath(), getOutputPath(), start, end);
  }

}
