package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

public class CVB0Config {

  public static final String NUM_TOPICS = "num_topics";
  public static final String NUM_TERMS = "num_terms";
  public static final String DOC_TOPIC_SMOOTHING = "doc_topic_smoothing";
  public static final String TERM_TOPIC_SMOOTHING = "term_topic_smoothing";
  public static final String DICTIONARY = "dictionary";
  public static final String DOC_TOPIC_OUTPUT = "doc_topic_output";
  public static final String MODEL_TEMP_DIR = "topic_model_temp_dir";
  public static final String ITERATION_BLOCK_SIZE = "iteration_block_size";
  public static final String RANDOM_SEED = "random_seed";
  public static final String TEST_SET_FRACTION = "test_set_fraction";
  public static final String NUM_TRAIN_THREADS = "num_train_threads";
  public static final String NUM_UPDATE_THREADS = "num_update_threads";
  public static final String MAX_ITERATIONS_PER_DOC = "max_doc_topic_iters";
  public static final String MODEL_WEIGHT = "prev_iter_mult";
  public static final String NUM_REDUCE_TASKS = "num_reduce_tasks";
  public static final String BACKFILL_PERPLEXITY = "backfill_perplexity";
  static final String MODEL_PATHS = "mahout.lda.cvb.modelPath";

  public static Path[] getModelPaths(Configuration conf) {
    String[] modelPathNames = conf.getStrings(MODEL_PATHS);
    if (modelPathNames == null || modelPathNames.length == 0) {
      return null;
    }
    Path[] modelPaths = new Path[modelPathNames.length];
    for (int i = 0; i < modelPathNames.length; i++) {
      modelPaths[i] = new Path(modelPathNames[i]);
    }
    return modelPaths;
  }


}
