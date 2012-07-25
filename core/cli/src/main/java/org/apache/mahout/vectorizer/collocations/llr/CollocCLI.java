package org.apache.mahout.vectorizer.collocations.llr;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CollocCLI extends AbstractCLI {
  public static final Logger log = LoggerFactory.getLogger(CollocCLI.class);

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new CollocCLI(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.numReducersOption().create());

    addOption("maxNGramSize",
              "ng",
              "(Optional) The max size of ngrams to create (2 = bigrams, 3 = trigrams, etc) default: 2",
              String.valueOf(CollocDriver.DEFAULT_MAX_NGRAM_SIZE));
    addOption("minSupport", "s", "(Optional) Minimum Support. Default Value: "
        + CollocReducer.DEFAULT_MIN_SUPPORT, String.valueOf(CollocReducer.DEFAULT_MIN_SUPPORT));
    addOption("minLLR", "ml", "(Optional)The minimum Log Likelihood Ratio(Float)  Default is "
        + LLRReducer.DEFAULT_MIN_LLR, String.valueOf(LLRReducer.DEFAULT_MIN_LLR));
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption("analyzerName", "a", "The class name of the analyzer to use for preprocessing", null);

    addFlag("preprocess", "p", "If set, input is SequenceFile<Text,Text> where the value is the document, "
        + " which will be tokenized using the specified analyzer.");
    addFlag("unigram", "u", "If set, unigrams will be emitted in the final output alongside collocations");

    Map<String, List<String>> argMap = parseArguments(args);

    if (argMap == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();

    int maxNGramSize = CollocDriver.DEFAULT_MAX_NGRAM_SIZE;
    if (hasOption("maxNGramSize")) {
      try {
        maxNGramSize = Integer.parseInt(getOption("maxNGramSize"));
      } catch (NumberFormatException ex) {
        log.warn("Could not parse ngram size option");
      }
    }
    log.info("Maximum n-gram size is: {}", maxNGramSize);

    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }

    int minSupport = CollocReducer.DEFAULT_MIN_SUPPORT;
    if (getOption("minSupport") != null) {
      minSupport = Integer.parseInt(getOption("minSupport"));
    }
    log.info("Minimum Support value: {}", minSupport);

    float minLLRValue = LLRReducer.DEFAULT_MIN_LLR;
    if (getOption("minLLR") != null) {
      minLLRValue = Float.parseFloat(getOption("minLLR"));
    }
    log.info("Minimum LLR value: {}", minLLRValue);

    int reduceTasks = CollocDriver.DEFAULT_PASS1_NUM_REDUCE_TASKS;
    if (getOption("maxRed") != null) {
      reduceTasks = Integer.parseInt(getOption("maxRed"));
    }
    log.info("Number of pass1 reduce tasks: {}", reduceTasks);

    boolean emitUnigrams = argMap.containsKey("emitUnigrams");

    CollocDriver driver = new CollocDriver();
    return driver.run(argMap.containsKey("preprocess"), getOption("analyzerName"), input, output, emitUnigrams, maxNGramSize, reduceTasks, minSupport, minLLRValue);
  }
}
