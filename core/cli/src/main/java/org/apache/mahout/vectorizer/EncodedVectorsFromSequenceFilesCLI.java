package org.apache.mahout.vectorizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.LuceneTextValueEncoder;

public class EncodedVectorsFromSequenceFilesCLI extends
    AbstractCLI {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new EncodedVectorsFromSequenceFilesCLI(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.analyzerOption().create());
    addOption(buildOption("sequentialAccessVector", "seq", "(Optional) Whether output vectors should be SequentialAccessVectors. If set true else false", false, false, null));
    addOption(buildOption("namedVector", "nv", "Create named vectors using the key.  False by default", false, false, null));
    addOption("cardinality", "c", "The cardinality to use for creating the vectors.  Default is 5000", String.valueOf(5000));
    addOption("encoderFieldName", "en", "The name of the encoder to be passed to the FeatureVectorEncoder constructor.  Default is text.  Note this is not the class name of a FeatureValueEncoder, but is instead the construction argument.", "text");
    addOption("encoderClass", "ec", "The class name of the encoder to be used. Default is " + LuceneTextValueEncoder.class.getName(), LuceneTextValueEncoder.class.getName());
    addOption(DefaultOptionCreator.overwriteOption().create());
    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();

    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }

    Class<? extends Analyzer> analyzerClass = getAnalyzerClassFromOption();


    Configuration conf = getConf();

    boolean sequentialAccessOutput = hasOption("sequentialAccessVector");


    boolean namedVectors = hasOption("namedVector");
    int cardinality = 5000;
    if (hasOption("cardinality")) {
      cardinality = Integer.parseInt(getOption("cardinality"));
    }
    String encoderName = "text";
    if (hasOption("encoderFieldName")) {
      encoderName = getOption("encoderFieldName");
    }
    String encoderClass = LuceneTextValueEncoder.class.getName();
    if (hasOption("encoderClass")) {
      encoderClass = getOption("encoderClass");
      ClassUtils.instantiateAs(encoderClass, FeatureVectorEncoder.class, new Class[]{String.class}, new Object[]{encoderName}); //try instantiating it
    }

    EncodedVectorsFromSequenceFiles job = new EncodedVectorsFromSequenceFiles();
    job.setConf(conf);
    return job.run(input, output, analyzerClass.getName(), encoderClass, encoderName, sequentialAccessOutput, namedVectors, cardinality);
  }
}
