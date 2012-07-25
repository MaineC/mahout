package org.apache.mahout.clustering.lda.cvb;

import static org.apache.mahout.clustering.lda.cvb.CVB0Config.BACKFILL_PERPLEXITY;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.DICTIONARY;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.DOC_TOPIC_OUTPUT;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.DOC_TOPIC_SMOOTHING;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.ITERATION_BLOCK_SIZE;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.MAX_ITERATIONS_PER_DOC;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.MODEL_TEMP_DIR;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.NUM_REDUCE_TASKS;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.NUM_TERMS;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.NUM_TOPICS;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.NUM_TRAIN_THREADS;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.NUM_UPDATE_THREADS;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.RANDOM_SEED;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.TERM_TOPIC_SMOOTHING;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.TEST_SET_FRACTION;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

public class CVB0CLI extends AbstractCLI {
  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION, "cd", "The convergence delta value", "0");
    addOption(DefaultOptionCreator.overwriteOption().create());

    addOption(NUM_TOPICS, "k", "Number of topics to learn", true);
    addOption(NUM_TERMS, "nt", "Vocabulary size", false);
    addOption(DOC_TOPIC_SMOOTHING, "a", "Smoothing for document/topic distribution", "0.0001");
    addOption(TERM_TOPIC_SMOOTHING, "e", "Smoothing for topic/term distribution", "0.0001");
    addOption(DICTIONARY, "dict", "Path to term-dictionary file(s) (glob expression supported)",
        false);
    addOption(DOC_TOPIC_OUTPUT, "dt", "Output path for the training doc/topic distribution",
        false);
    addOption(MODEL_TEMP_DIR, "mt", "Path to intermediate model path (useful for restarting)",
        false);
    addOption(ITERATION_BLOCK_SIZE, "block", "Number of iterations per perplexity check", "10");
    addOption(RANDOM_SEED, "seed", "Random seed", false);
    addOption(TEST_SET_FRACTION, "tf", "Fraction of data to hold out for testing", "0");
    addOption(NUM_TRAIN_THREADS, "ntt", "number of threads per mapper to train with", "4");
    addOption(NUM_UPDATE_THREADS, "nut", "number of threads per mapper to update the model with",
        "1");
    addOption(MAX_ITERATIONS_PER_DOC, "mipd",
        "max number of iterations per doc for p(topic|doc) learning", "10");
    addOption(NUM_REDUCE_TASKS, null,
        "number of reducers to use during model estimation", "10");
    addOption(buildOption(BACKFILL_PERPLEXITY, null,
        "enable backfilling of missing perplexity values", false, false, null));

    if (parseArguments(args) == null) {
      return -1;
    }

    int numTopics = Integer.parseInt(getOption(NUM_TOPICS));
    Path inputPath = getInputPath();
    Path topicModelOutputPath = getOutputPath();
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    int iterationBlockSize = Integer.parseInt(getOption(ITERATION_BLOCK_SIZE));
    double convergenceDelta = Double.parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    double alpha = Double.parseDouble(getOption(DOC_TOPIC_SMOOTHING));
    double eta = Double.parseDouble(getOption(TERM_TOPIC_SMOOTHING));
    int numTrainThreads = Integer.parseInt(getOption(NUM_TRAIN_THREADS));
    int numUpdateThreads = Integer.parseInt(getOption(NUM_UPDATE_THREADS));
    int maxItersPerDoc = Integer.parseInt(getOption(MAX_ITERATIONS_PER_DOC));
    Path dictionaryPath = hasOption(DICTIONARY) ? new Path(getOption(DICTIONARY)) : null;
    int numTerms = hasOption(NUM_TERMS)
                 ? Integer.parseInt(getOption(NUM_TERMS))
                 : getNumTerms(getConf(), dictionaryPath);
    Path docTopicOutputPath = hasOption(DOC_TOPIC_OUTPUT) ? new Path(getOption(DOC_TOPIC_OUTPUT)) : null;
    Path modelTempPath = hasOption(MODEL_TEMP_DIR)
                       ? new Path(getOption(MODEL_TEMP_DIR))
                       : getTempPath("topicModelState");
    long seed = hasOption(RANDOM_SEED)
              ? Long.parseLong(getOption(RANDOM_SEED))
              : System.nanoTime() % 10000;
    float testFraction = hasOption(TEST_SET_FRACTION)
                       ? Float.parseFloat(getOption(TEST_SET_FRACTION))
                       : 0.0f;
    int numReduceTasks = Integer.parseInt(getOption(NUM_REDUCE_TASKS));
    boolean backfillPerplexity = hasOption(BACKFILL_PERPLEXITY);

    CVB0Driver job = new CVB0Driver();
    return job.run(getConf(), inputPath, topicModelOutputPath, numTopics, numTerms, alpha, eta,
        maxIterations, iterationBlockSize, convergenceDelta, dictionaryPath, docTopicOutputPath,
        modelTempPath, seed, testFraction, numTrainThreads, numUpdateThreads, maxItersPerDoc,
        numReduceTasks, backfillPerplexity);
  }
  private static int getNumTerms(Configuration conf, Path dictionaryPath) throws IOException {
    FileSystem fs = dictionaryPath.getFileSystem(conf);
    Text key = new Text();
    IntWritable value = new IntWritable();
    int maxTermId = -1;
    for (FileStatus stat : fs.globStatus(dictionaryPath)) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, stat.getPath(), conf);
      while (reader.next(key, value)) {
        maxTermId = Math.max(maxTermId, value.get());
      }
    }
    return maxTermId + 1;
  }


  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new CVB0CLI(), args);
  }

}
