package org.apache.mahout.clustering.lda.cvb;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class InMemoryCollapsedVariationalBayes0CLI extends
    AbstractCLI {

  private static final Logger log = LoggerFactory.getLogger(InMemoryCollapsedVariationalBayes0CLI.class);

  
  public static int main2(String[] args, Configuration conf) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option helpOpt = DefaultOptionCreator.helpOption();

    Option inputDirOpt = obuilder.withLongName("input").withRequired(true).withArgument(
      abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
      "The Directory on HDFS containing the collapsed, properly formatted files having "
          + "one doc per line").withShortName("i").create();

    Option dictOpt = obuilder.withLongName("dictionary").withRequired(false).withArgument(
      abuilder.withName("dictionary").withMinimum(1).withMaximum(1).create()).withDescription(
      "The path to the term-dictionary format is ... ").withShortName("d").create();

    Option dfsOpt = obuilder.withLongName("dfs").withRequired(false).withArgument(
      abuilder.withName("dfs").withMinimum(1).withMaximum(1).create()).withDescription(
      "HDFS namenode URI").withShortName("dfs").create();

    Option numTopicsOpt = obuilder.withLongName("numTopics").withRequired(true).withArgument(abuilder
        .withName("numTopics").withMinimum(1).withMaximum(1)
        .create()).withDescription("Number of topics to learn").withShortName("top").create();

    Option outputTopicFileOpt = obuilder.withLongName("topicOutputFile").withRequired(true).withArgument(
        abuilder.withName("topicOutputFile").withMinimum(1).withMaximum(1).create())
        .withDescription("File to write out p(term | topic)").withShortName("to").create();

    Option outputDocFileOpt = obuilder.withLongName("docOutputFile").withRequired(true).withArgument(
        abuilder.withName("docOutputFile").withMinimum(1).withMaximum(1).create())
        .withDescription("File to write out p(topic | docid)").withShortName("do").create();

    Option alphaOpt = obuilder.withLongName("alpha").withRequired(false).withArgument(abuilder
        .withName("alpha").withMinimum(1).withMaximum(1).withDefault("0.1").create())
        .withDescription("Smoothing parameter for p(topic | document) prior").withShortName("a").create();

    Option etaOpt = obuilder.withLongName("eta").withRequired(false).withArgument(abuilder
        .withName("eta").withMinimum(1).withMaximum(1).withDefault("0.1").create())
        .withDescription("Smoothing parameter for p(term | topic)").withShortName("e").create();

    Option maxIterOpt = obuilder.withLongName("maxIterations").withRequired(false).withArgument(abuilder
        .withName("maxIterations").withMinimum(1).withMaximum(1).withDefault(10).create())
        .withDescription("Maximum number of training passes").withShortName("m").create();

    Option modelCorpusFractionOption = obuilder.withLongName("modelCorpusFraction")
        .withRequired(false).withArgument(abuilder.withName("modelCorpusFraction").withMinimum(1)
        .withMaximum(1).withDefault(0.0).create()).withShortName("mcf")
        .withDescription("For online updates, initial value of |model|/|corpus|").create();

    Option burnInOpt = obuilder.withLongName("burnInIterations").withRequired(false).withArgument(abuilder
        .withName("burnInIterations").withMinimum(1).withMaximum(1).withDefault(5).create())
        .withDescription("Minimum number of iterations").withShortName("b").create();

    Option convergenceOpt = obuilder.withLongName("convergence").withRequired(false).withArgument(abuilder
        .withName("convergence").withMinimum(1).withMaximum(1).withDefault("0.0").create())
        .withDescription("Fractional rate of perplexity to consider convergence").withShortName("c").create();

    Option reInferDocTopicsOpt = obuilder.withLongName("reInferDocTopics").withRequired(false)
        .withArgument(abuilder.withName("reInferDocTopics").withMinimum(1).withMaximum(1)
        .withDefault("no").create())
        .withDescription("re-infer p(topic | doc) : [no | randstart | continue]")
        .withShortName("rdt").create();

    Option numTrainThreadsOpt = obuilder.withLongName("numTrainThreads").withRequired(false)
        .withArgument(abuilder.withName("numTrainThreads").withMinimum(1).withMaximum(1)
        .withDefault("1").create())
        .withDescription("number of threads to train with")
        .withShortName("ntt").create();

    Option numUpdateThreadsOpt = obuilder.withLongName("numUpdateThreads").withRequired(false)
        .withArgument(abuilder.withName("numUpdateThreads").withMinimum(1).withMaximum(1)
        .withDefault("1").create())
        .withDescription("number of threads to update the model with")
        .withShortName("nut").create();

    Option verboseOpt = obuilder.withLongName("verbose").withRequired(false)
        .withArgument(abuilder.withName("verbose").withMinimum(1).withMaximum(1)
        .withDefault("false").create())
        .withDescription("print verbose information, like top-terms in each topic, during iteration")
        .withShortName("v").create();

    Group group = gbuilder.withName("Options").withOption(inputDirOpt).withOption(numTopicsOpt)
        .withOption(alphaOpt).withOption(etaOpt)
        .withOption(maxIterOpt).withOption(burnInOpt).withOption(convergenceOpt)
        .withOption(dictOpt).withOption(reInferDocTopicsOpt)
        .withOption(outputDocFileOpt).withOption(outputTopicFileOpt).withOption(dfsOpt)
        .withOption(numTrainThreadsOpt).withOption(numUpdateThreadsOpt)
        .withOption(modelCorpusFractionOption).withOption(verboseOpt).create();

    try {
      Parser parser = new Parser();

      parser.setGroup(group);
      parser.setHelpOption(helpOpt);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return -1;
      }

      String inputDirString = (String) cmdLine.getValue(inputDirOpt);
      String dictDirString = cmdLine.hasOption(dictOpt) ? (String)cmdLine.getValue(dictOpt) : null;
      int numTopics = Integer.parseInt((String) cmdLine.getValue(numTopicsOpt));
      double alpha = Double.parseDouble((String)cmdLine.getValue(alphaOpt));
      double eta = Double.parseDouble((String)cmdLine.getValue(etaOpt));
      int maxIterations = Integer.parseInt((String)cmdLine.getValue(maxIterOpt));
      int burnInIterations = (Integer)cmdLine.getValue(burnInOpt);
      double minFractionalErrorChange = Double.parseDouble((String) cmdLine.getValue(convergenceOpt));
      int numTrainThreads = Integer.parseInt((String)cmdLine.getValue(numTrainThreadsOpt));
      int numUpdateThreads = Integer.parseInt((String)cmdLine.getValue(numUpdateThreadsOpt));
      String topicOutFile = (String)cmdLine.getValue(outputTopicFileOpt);
      String docOutFile = (String)cmdLine.getValue(outputDocFileOpt);
      String reInferDocTopics = (String)cmdLine.getValue(reInferDocTopicsOpt);
      boolean verbose = Boolean.parseBoolean((String) cmdLine.getValue(verboseOpt));
      double modelCorpusFraction = (Double) cmdLine.getValue(modelCorpusFractionOption);

      String dfsNameNode = (String)cmdLine.getValue(dfsOpt);
      
      return InMemoryCollapsedVariationalBayes0.run(conf, dfsNameNode, dictDirString, inputDirString,
          numTrainThreads, numTopics, alpha, eta, numUpdateThreads, verbose, minFractionalErrorChange,
          maxIterations, burnInIterations, modelCorpusFraction, reInferDocTopics, topicOutFile, docOutFile);
    } catch (OptionException e) {
      log.error("Error while parsing options", e);
      CommandLineUtil.printHelp(group);
    }
    return -1;

  }

  @Override
  public int run(String[] strings) throws Exception {
    return main2(strings, getConf());
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new InMemoryCollapsedVariationalBayes0CLI(), args);
  }

}
