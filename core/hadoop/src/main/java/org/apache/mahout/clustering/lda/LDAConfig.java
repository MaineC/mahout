package org.apache.mahout.clustering.lda;

import java.util.Arrays;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.mahout.common.IntPairWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.DenseMatrix;

import com.google.common.base.Preconditions;

public class LDAConfig {
  static final String STATE_IN_KEY = "org.apache.mahout.clustering.lda.stateIn";
  static final String NUM_TOPICS_KEY = "org.apache.mahout.clustering.lda.numTopics";
  static final String NUM_WORDS_KEY = "org.apache.mahout.clustering.lda.numWords";
  static final String TOPIC_SMOOTHING_KEY = "org.apache.mahout.clustering.lda.topicSmoothing";
  static final int LOG_LIKELIHOOD_KEY = -2;
  static final int TOPIC_SUM_KEY = -1;
  static final double OVERALL_CONVERGENCE = 1.0E-5;

  public static LDAState createState(Configuration job) {
    return createState(job, false);
  }


  public static LDAState createState(Configuration job, boolean empty) {
    String statePath = job.get(STATE_IN_KEY);
    int numTopics = Integer.parseInt(job.get(NUM_TOPICS_KEY));
    int numWords = Integer.parseInt(job.get(NUM_WORDS_KEY));
    double topicSmoothing = Double.parseDouble(job.get(TOPIC_SMOOTHING_KEY));

    Path dir = new Path(statePath);

    // TODO scalability bottleneck: numWords * numTopics * 8bytes for the driver *and* M/R classes
    DenseMatrix pWgT = new DenseMatrix(numTopics, numWords);
    double[] logTotals = new double[numTopics];
    Arrays.fill(logTotals, Double.NEGATIVE_INFINITY);
    double ll = 0.0;
    if (empty) {
      return new LDAState(numTopics, numWords, topicSmoothing, pWgT, logTotals, ll);
    }
    for (Pair<IntPairWritable,DoubleWritable> record
         : new SequenceFileDirIterable<IntPairWritable, DoubleWritable>(new Path(dir, "part-*"),
                                                                        PathType.GLOB,
                                                                        null,
                                                                        null,
                                                                        true,
                                                                        job)) {
      IntPairWritable key = record.getFirst();
      DoubleWritable value = record.getSecond();
      int topic = key.getFirst();
      int word = key.getSecond();
      if (word == TOPIC_SUM_KEY) {
        logTotals[topic] = value.get();
        Preconditions.checkArgument(!Double.isInfinite(value.get()));
      } else if (topic == LOG_LIKELIHOOD_KEY) {
        ll = value.get();
      } else {
        Preconditions.checkArgument(topic >= 0, "topic should be non-negative, not %d", topic);
        Preconditions.checkArgument(word >= 0, "word should be non-negative not %d", word);
        Preconditions.checkArgument(pWgT.getQuick(topic, word) == 0.0);

        pWgT.setQuick(topic, word, value.get());
        Preconditions.checkArgument(!Double.isInfinite(pWgT.getQuick(topic, word)));
      }
    }

    return new LDAState(numTopics, numWords, topicSmoothing, pWgT, logTotals, ll);
  }

}
