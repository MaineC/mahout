/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.clustering.lda.cvb;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.mapreduce.AbstractJob;
import org.apache.mahout.common.mapreduce.VectorSumReducer;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.util.List;
import static org.apache.mahout.clustering.lda.cvb.CVB0Config.*;

/**
 * See {@link CachingCVB0Mapper} for more details on scalability and room for improvement.
 * To try out this LDA implementation without using Hadoop, check out
 * {@link InMemoryCollapsedVariationalBayes0}.  If you want to do training directly in java code
 * with your own main(), then look to {@link ModelTrainer} and {@link TopicModel}.
 *
 * Usage: {@code ./bin/mahout cvb <i>options</i>}
 * <p>
 * Valid options include:
 * <dl>
 * <dt>{@code --input path}</td>
 * <dd>Input path for {@code SequenceFile<IntWritable, VectorWritable>} document vectors. See
 * {@link SparseVectorsFromSequenceFiles} for details on how to generate this input format.</dd>
 * <dt>{@code --dictionary path}</dt>
 * <dd>Path to dictionary file(s) generated during construction of input document vectors (glob
 * expression supported). If set, this data is scanned to determine an appropriate value for option
 * {@code --num_terms}.</dd>
 * <dt>{@code --output path}</dt>
 * <dd>Output path for topic-term distributions.</dd>
 * <dt>{@code --doc_topic_output path}</dt>
 * <dd>Output path for doc-topic distributions.</dd>
 * <dt>{@code --num_topics k}</dt>
 * <dd>Number of latent topics.</dd>
 * <dt>{@code --num_terms nt}</dt>
 * <dd>Number of unique features defined by input document vectors. If option {@code --dictionary}
 * is defined and this option is unspecified, term count is calculated from dictionary.</dd>
 * <dt>{@code --topic_model_temp_dir path}</dt>
 * <dd>Path in which to store model state after each iteration.</dd>
 * <dt>{@code --maxIter i}</dt>
 * <dd>Maximum number of iterations to perform. If this value is less than or equal to the number of
 * iteration states found beneath the path specified by option {@code --topic_model_temp_dir}, no
 * further iterations are performed. Instead, output topic-term and doc-topic distributions are
 * generated using data from the specified iteration.</dd>
 * <dt>{@code --max_doc_topic_iters i}</dt>
 * <dd>Maximum number of iterations per doc for p(topic|doc) learning. Defaults to {@code 10}.</dd>
 * <dt>{@code --doc_topic_smoothing a}</dt>
 * <dd>Smoothing for doc-topic distribution. Defaults to {@code 0.0001}.</dd>
 * <dt>{@code --term_topic_smoothing e}</dt>
 * <dd>Smoothing for topic-term distribution. Defaults to {@code 0.0001}.</dd>
 * <dt>{@code --random_seed seed}</dt>
 * <dd>Integer seed for random number generation.</dd>
 * <dt>{@code --test_set_percentage p}</dt>
 * <dd>Fraction of data to hold out for testing. Defaults to {@code 0.0}.</dd>
 * <dt>{@code --iteration_block_size block}</dt>
 * <dd>Number of iterations between perplexity checks. Defaults to {@code 10}. This option is
 * ignored unless option {@code --test_set_percentage} is greater than zero.</dd>
 * </dl>
 */
public class CVB0Driver extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(CVB0Driver.class);

  public static int run(Configuration conf,
                        Path inputPath,
                        Path topicModelOutputPath,
                        int numTopics,
                        int numTerms,
                        double alpha,
                        double eta,
                        int maxIterations,
                        int iterationBlockSize,
                        double convergenceDelta,
                        Path dictionaryPath,
                        Path docTopicOutputPath,
                        Path topicModelStateTempPath,
                        long randomSeed,
                        float testFraction,
                        int numTrainThreads,
                        int numUpdateThreads,
                        int maxItersPerDoc,
                        int numReduceTasks,
                        boolean backfillPerplexity)
      throws ClassNotFoundException, IOException, InterruptedException {
    // verify arguments
    Preconditions.checkArgument(testFraction >= 0.0 && testFraction <= 1.0,
        "Expected 'testFraction' value in range [0, 1] but found value '%s'", testFraction);
    Preconditions.checkArgument(!backfillPerplexity || testFraction > 0.0,
        "Expected 'testFraction' value in range (0, 1] but found value '%s'", testFraction);

    String infoString = "Will run Collapsed Variational Bayes (0th-derivative approximation) " 
      + "learning for LDA on {} (numTerms: {}), finding {}-topics, with document/topic prior {}, " 
      + "topic/term prior {}.  Maximum iterations to run will be {}, unless the change in " 
      + "perplexity is less than {}.  Topic model output (p(term|topic) for each topic) will be " 
      + "stored {}.  Random initialization seed is {}, holding out {} of the data for perplexity " 
      + "check\n";
    log.info(infoString, new Object[] {inputPath, numTerms, numTopics, alpha, eta, maxIterations,
        convergenceDelta, topicModelOutputPath, randomSeed, testFraction});
    infoString = dictionaryPath == null
               ? "" : "Dictionary to be used located " + dictionaryPath.toString() + '\n';
    infoString += docTopicOutputPath == null
               ? "" : "p(topic|docId) will be stored " + docTopicOutputPath.toString() + '\n';
    log.info(infoString);

    FileSystem fs = FileSystem.get(topicModelStateTempPath.toUri(), conf);
    int iterationNumber = getCurrentIterationNumber(conf, topicModelStateTempPath, maxIterations);
    log.info("Current iteration number: {}", iterationNumber);

    conf.set(NUM_TOPICS, String.valueOf(numTopics));
    conf.set(NUM_TERMS, String.valueOf(numTerms));
    conf.set(DOC_TOPIC_SMOOTHING, String.valueOf(alpha));
    conf.set(TERM_TOPIC_SMOOTHING, String.valueOf(eta));
    conf.set(RANDOM_SEED, String.valueOf(randomSeed));
    conf.set(NUM_TRAIN_THREADS, String.valueOf(numTrainThreads));
    conf.set(NUM_UPDATE_THREADS, String.valueOf(numUpdateThreads));
    conf.set(MAX_ITERATIONS_PER_DOC, String.valueOf(maxItersPerDoc));
    conf.set(MODEL_WEIGHT, "1"); // TODO
    conf.set(TEST_SET_FRACTION, String.valueOf(testFraction));

    List<Double> perplexities = Lists.newArrayList();
    for (int i = 1; i <= iterationNumber; i++) {
      // form path to model
      Path modelPath = modelPath(topicModelStateTempPath, i);

      // read perplexity
      double perplexity = readPerplexity(conf, topicModelStateTempPath, i);
      if (Double.isNaN(perplexity)) {
        if (!(backfillPerplexity && i % iterationBlockSize == 0)) {
          continue;
        }
        log.info("Backfilling perplexity at iteration {}", i);
        if (!fs.exists(modelPath)) {
          log.error("Model path '{}' does not exist; Skipping iteration {} perplexity calculation", modelPath.toString(), i);
          continue;
        }
        perplexity = calculatePerplexity(conf, inputPath, modelPath, i);
      }

      // register and log perplexity
      perplexities.add(perplexity);
      log.info("Perplexity at iteration {} = {}", i, perplexity);
    }

    long startTime = System.currentTimeMillis();
    while (iterationNumber < maxIterations) {
      // test convergence
      if (convergenceDelta > 0.0) {
        double delta = rateOfChange(perplexities);
        if (delta < convergenceDelta) {
          log.info("Convergence achieved at iteration {} with perplexity {} and delta {}",
              new Object[]{iterationNumber, perplexities.get(perplexities.size() - 1), delta});
          break;
        }
      }

      // update model
      iterationNumber++;
      log.info("About to run iteration {} of {}", iterationNumber, maxIterations);
      Path modelInputPath = modelPath(topicModelStateTempPath, iterationNumber - 1);
      Path modelOutputPath = modelPath(topicModelStateTempPath, iterationNumber);
      runIteration(conf, inputPath, modelInputPath, modelOutputPath, iterationNumber,
          maxIterations, numReduceTasks);

      // calculate perplexity
      if (testFraction > 0 && iterationNumber % iterationBlockSize == 0) {
        perplexities.add(calculatePerplexity(conf, inputPath, modelOutputPath, iterationNumber));
        log.info("Current perplexity = {}", perplexities.get(perplexities.size() - 1));
        log.info("(p_{} - p_{}) / p_0 = {}; target = {}", new Object[]{
            iterationNumber , iterationNumber - iterationBlockSize, rateOfChange(perplexities), convergenceDelta
        });
      }
    }
    log.info("Completed {} iterations in {} seconds", iterationNumber,
        (System.currentTimeMillis() - startTime)/1000);
    log.info("Perplexities: ({})", Joiner.on(", ").join(perplexities));

    // write final topic-term and doc-topic distributions
    Path finalIterationData = modelPath(topicModelStateTempPath, iterationNumber);
    Job topicModelOutputJob = topicModelOutputPath != null
        ? writeTopicModel(conf, finalIterationData, topicModelOutputPath)
        : null;
    Job docInferenceJob = docTopicOutputPath != null
        ? writeDocTopicInference(conf, inputPath, finalIterationData, docTopicOutputPath)
        : null;
    if (topicModelOutputJob != null && !topicModelOutputJob.waitForCompletion(true)) {
      return -1;
    }
    if (docInferenceJob != null && !docInferenceJob.waitForCompletion(true)) {
      return -1;
    }
    return 0;
  }

  private static double rateOfChange(List<Double> perplexities) {
    int sz = perplexities.size();
    if (sz < 2) {
      return Double.MAX_VALUE;
    }
    return Math.abs(perplexities.get(sz - 1) - perplexities.get(sz - 2)) / perplexities.get(0);
  }

  private static double calculatePerplexity(Configuration conf, Path corpusPath, Path modelPath, int iteration)
      throws IOException,
      ClassNotFoundException, InterruptedException {
    String jobName = "Calculating perplexity for " + modelPath;
    log.info("About to run: " + jobName);
    Job job = new Job(conf, jobName);
    job.setJarByClass(CachingCVB0PerplexityMapper.class);
    job.setMapperClass(CachingCVB0PerplexityMapper.class);
    job.setCombinerClass(DualDoubleSumReducer.class);
    job.setReducerClass(DualDoubleSumReducer.class);
    job.setNumReduceTasks(1);
    job.setOutputKeyClass(DoubleWritable.class);
    job.setOutputValueClass(DoubleWritable.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.addInputPath(job, corpusPath);
    Path outputPath = perplexityPath(modelPath.getParent(), iteration);
    FileOutputFormat.setOutputPath(job, outputPath);
    setModelPaths(job, modelPath);
    HadoopUtil.delete(conf, outputPath);
    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("Failed to calculate perplexity for: " + modelPath);
    }
    return readPerplexity(conf, modelPath.getParent(), iteration);
  }

  /**
   * Sums keys and values independently.
   */
  public static class DualDoubleSumReducer extends
    Reducer<DoubleWritable, DoubleWritable, DoubleWritable, DoubleWritable> {
    private final DoubleWritable outKey = new DoubleWritable();
    private final DoubleWritable outValue = new DoubleWritable();

    @Override
    public void run(Context context) throws IOException,
        InterruptedException {
      double keySum = 0.0;
      double valueSum = 0.0;
      while (context.nextKey()) {
        keySum += context.getCurrentKey().get();
        for (DoubleWritable value : context.getValues()) {
          valueSum += value.get();
        }
      }
      outKey.set(keySum);
      outValue.set(valueSum);
      context.write(outKey, outValue);
    }
  }

  /**
   * @param topicModelStateTemp
   * @param iteration
   * @return {@code double[2]} where first value is perplexity and second is model weight of those
   *         documents sampled during perplexity computation, or {@code null} if no perplexity data
   *         exists for the given iteration.
   * @throws IOException
   */
  public static double readPerplexity(Configuration conf, Path topicModelStateTemp, int iteration)
      throws IOException {
    Path perplexityPath = perplexityPath(topicModelStateTemp, iteration);
    FileSystem fs = FileSystem.get(perplexityPath.toUri(), conf);
    if (!fs.exists(perplexityPath)) {
      log.warn("Perplexity path {} does not exist, returning NaN", perplexityPath);
      return Double.NaN;
    }
    double perplexity = 0;
    double modelWeight = 0;
    long n = 0;
    for (Pair<DoubleWritable, DoubleWritable> pair : new SequenceFileDirIterable<DoubleWritable, DoubleWritable>(
        perplexityPath, PathType.LIST, PathFilters.partFilter(), null, true, conf)) {
      modelWeight += pair.getFirst().get();
      perplexity += pair.getSecond().get();
      n++;
    }
    log.info("Read {} entries with total perplexity {} and model weight {}", new Object[] { n,
            perplexity, modelWeight });
    return perplexity / modelWeight;
  }

  private static Job writeTopicModel(Configuration conf, Path modelInput, Path output) throws IOException,
      InterruptedException, ClassNotFoundException {
    String jobName = String.format("Writing final topic/term distributions from %s to %s", modelInput,
        output);
    log.info("About to run: " + jobName);
    Job job = new Job(conf, jobName);
    job.setJarByClass(CVB0Driver.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setMapperClass(CVB0TopicTermVectorNormalizerMapper.class);
    job.setNumReduceTasks(0);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.addInputPath(job, modelInput);
    FileOutputFormat.setOutputPath(job, output);
    job.submit();
    return job;
  }

  private static Job writeDocTopicInference(Configuration conf, Path corpus, Path modelInput, Path output)
      throws IOException, ClassNotFoundException, InterruptedException {
    String jobName = String.format("Writing final document/topic inference from %s to %s", corpus,
        output);
    log.info("About to run: " + jobName);
    Job job = new Job(conf, jobName);
    job.setMapperClass(CVB0DocInferenceMapper.class);
    job.setNumReduceTasks(0);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    FileSystem fs = FileSystem.get(corpus.toUri(), conf);
    if (modelInput != null && fs.exists(modelInput)) {
      FileStatus[] statuses = fs.listStatus(modelInput, PathFilters.partFilter());
      URI[] modelUris = new URI[statuses.length];
      for (int i = 0; i < statuses.length; i++) {
        modelUris[i] = statuses[i].getPath().toUri();
      }
      DistributedCache.setCacheFiles(modelUris, conf);
    }
    FileInputFormat.addInputPath(job, corpus);
    FileOutputFormat.setOutputPath(job, output);
    job.setJarByClass(CVB0Driver.class);
    job.submit();
    return job;
  }

  public static Path modelPath(Path topicModelStateTempPath, int iterationNumber) {
    return new Path(topicModelStateTempPath, "model-" + iterationNumber);
  }

  public static Path stage1OutputPath(Path topicModelStateTempPath, int iterationNumber) {
    return new Path(topicModelStateTempPath, "tmp-" + iterationNumber);
  }

  public static Path perplexityPath(Path topicModelStateTempPath, int iterationNumber) {
    return new Path(topicModelStateTempPath, "perplexity-" + iterationNumber);
  }

  private static int getCurrentIterationNumber(Configuration config, Path modelTempDir, int maxIterations)
      throws IOException {
    FileSystem fs = FileSystem.get(modelTempDir.toUri(), config);
    int iterationNumber = 1;
    Path iterationPath = modelPath(modelTempDir, iterationNumber);
    while (fs.exists(iterationPath) && iterationNumber <= maxIterations) {
      log.info("Found previous state: " + iterationPath);
      iterationNumber++;
      iterationPath = modelPath(modelTempDir, iterationNumber);
    }
    return iterationNumber - 1;
  }

  public static void runIteration(Configuration conf, Path corpusInput, Path modelInput, Path modelOutput,
                                  int iterationNumber, int maxIterations, int numReduceTasks) throws IOException, ClassNotFoundException, InterruptedException {
    String jobName = String.format("Iteration %d of %d, input path: %s",
        iterationNumber, maxIterations, modelInput);
    log.info("About to run: " + jobName);
    Job job = new Job(conf, jobName);
    job.setJarByClass(CVB0Driver.class);
    job.setMapperClass(CachingCVB0Mapper.class);
    job.setCombinerClass(VectorSumReducer.class);
    job.setReducerClass(VectorSumReducer.class);
    job.setNumReduceTasks(numReduceTasks);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.addInputPath(job, corpusInput);
    FileOutputFormat.setOutputPath(job, modelOutput);
    setModelPaths(job, modelInput);
    HadoopUtil.delete(conf, modelOutput);
    if (!job.waitForCompletion(true)) {
      throw new InterruptedException(String.format("Failed to complete iteration %d stage 1",
          iterationNumber));
    }
  }

  private static void setModelPaths(Job job, Path modelPath) throws IOException {
    Configuration conf = job.getConfiguration();
    if (modelPath == null || !FileSystem.get(modelPath.toUri(), conf).exists(modelPath)) {
      return;
    }
    FileStatus[] statuses = FileSystem.get(modelPath.toUri(), conf).listStatus(modelPath, PathFilters.partFilter());
    Preconditions.checkState(statuses.length > 0, "No part files found in model path '%s'", modelPath.toString());
    String[] modelPaths = new String[statuses.length];
    for (int i = 0; i < statuses.length; i++) {
      modelPaths[i] = statuses[i].getPath().toUri().toString();
    }
    conf.setStrings(CVB0Config.MODEL_PATHS, modelPaths);
  }
}
