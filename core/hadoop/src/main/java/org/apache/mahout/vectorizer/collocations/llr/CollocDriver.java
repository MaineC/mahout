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

package org.apache.mahout.vectorizer.collocations.llr;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.vectorizer.DefaultAnalyzer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.apache.mahout.vectorizer.collocations.llr.CollocConfig.*;

/** Driver for LLR Collocation discovery mapreduce job */
public final class CollocDriver extends Configured {
  //public static final String DEFAULT_OUTPUT_DIRECTORY = "output";
  public static final int DEFAULT_MAX_NGRAM_SIZE = 2;

  public static final int DEFAULT_PASS1_NUM_REDUCE_TASKS = 1;

  public static final Logger log = LoggerFactory.getLogger(CollocDriver.class);
  
  public int run(boolean preprocess, String className, Path input, Path output,
      boolean emitUnigrams, int maxNGramSize, int reduceTasks, int minSupport, float minLLRValue) throws ClassNotFoundException, IOException, InterruptedException {
    if (preprocess) {
      log.info("Input will be preprocessed");

      Class<? extends Analyzer> analyzerClass = DefaultAnalyzer.class;
      if (className != null) {
        analyzerClass = Class.forName(className).asSubclass(Analyzer.class);
        // try instantiating it, b/c there isn't any point in setting it if
        // you can't instantiate it
        ClassUtils.instantiateAs(analyzerClass, Analyzer.class);
      }

      Path tokenizedPath = new Path(output, DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);

      DocumentProcessor.tokenizeDocuments(input, analyzerClass, tokenizedPath, getConf());
      input = tokenizedPath;
    } else {
      log.info("Input will NOT be preprocessed");
    }

    // parse input and extract collocations
    long ngramCount =
      generateCollocations(input, output, getConf(), emitUnigrams, maxNGramSize, reduceTasks, minSupport);

    // tally collocations and perform LLR calculation
    computeNGramsPruneByLLR(output, getConf(), ngramCount, emitUnigrams, minLLRValue, reduceTasks);

    return 0;
  }

  /**
   * Generate all ngrams for the {@link org.apache.mahout.vectorizer.DictionaryVectorizer} job
   * 
   * @param input
   *          input path containing tokenized documents
   * @param output
   *          output path where ngrams are generated including unigrams
   * @param baseConf
   *          job configuration
   * @param maxNGramSize
   *          minValue = 2.
   * @param minSupport
   *          minimum support to prune ngrams including unigrams
   * @param minLLRValue
   *          minimum threshold to prune ngrams
   * @param reduceTasks
   *          number of reducers used
   */
  public static void generateAllGrams(Path input,
                                      Path output,
                                      Configuration baseConf,
                                      int maxNGramSize,
                                      int minSupport,
                                      float minLLRValue,
                                      int reduceTasks)
    throws IOException, InterruptedException, ClassNotFoundException {
    // parse input and extract collocations
    long ngramCount = generateCollocations(input, output, baseConf, true, maxNGramSize, reduceTasks, minSupport);

    // tally collocations and perform LLR calculation
    computeNGramsPruneByLLR(output, baseConf, ngramCount, true, minLLRValue, reduceTasks);
  }

  /**
   * pass1: generate collocations, ngrams
   */
  private static long generateCollocations(Path input,
                                           Path output,
                                           Configuration baseConf,
                                           boolean emitUnigrams,
                                           int maxNGramSize,
                                           int reduceTasks,
                                           int minSupport)
    throws IOException, ClassNotFoundException, InterruptedException {

    Configuration con = new Configuration(baseConf);
    con.setBoolean(EMIT_UNIGRAMS, emitUnigrams);
    con.setInt(CollocMapper.MAX_SHINGLE_SIZE, maxNGramSize);
    con.setInt(CollocReducer.MIN_SUPPORT, minSupport);
    
    Job job = new Job(con);
    job.setJobName(CollocDriver.class.getSimpleName() + ".generateCollocations:" + input);
    job.setJarByClass(CollocDriver.class);
    
    job.setMapOutputKeyClass(GramKey.class);
    job.setMapOutputValueClass(Gram.class);
    job.setPartitionerClass(GramKeyPartitioner.class);
    job.setGroupingComparatorClass(GramKeyGroupComparator.class);

    job.setOutputKeyClass(Gram.class);
    job.setOutputValueClass(Gram.class);

    job.setCombinerClass(CollocCombiner.class);

    FileInputFormat.setInputPaths(job, input);

    Path outputPath = new Path(output, SUBGRAM_OUTPUT_DIRECTORY);
    FileOutputFormat.setOutputPath(job, outputPath);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setMapperClass(CollocMapper.class);

    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setReducerClass(CollocReducer.class);
    job.setNumReduceTasks(reduceTasks);
    
    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) 
      throw new IllegalStateException("Job failed!");

    return job.getCounters().findCounter(CollocMapper.Count.NGRAM_TOTAL).getValue();
  }

  /**
   * pass2: perform the LLR calculation
   */
  private static void computeNGramsPruneByLLR(Path output,
                                              Configuration baseConf,
                                              long nGramTotal,
                                              boolean emitUnigrams,
                                              float minLLRValue,
                                              int reduceTasks)
    throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration(baseConf);
    conf.setLong(LLRReducer.NGRAM_TOTAL, nGramTotal);
    conf.setBoolean(EMIT_UNIGRAMS, emitUnigrams);
    conf.setFloat(LLRReducer.MIN_LLR, minLLRValue);

    Job job = new Job(conf);
    job.setJobName(CollocDriver.class.getSimpleName() + ".computeNGrams: " + output);
    job.setJarByClass(CollocDriver.class);
    
    job.setMapOutputKeyClass(Gram.class);
    job.setMapOutputValueClass(Gram.class);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(DoubleWritable.class);

    FileInputFormat.setInputPaths(job, new Path(output, SUBGRAM_OUTPUT_DIRECTORY));
    Path outPath = new Path(output, NGRAM_OUTPUT_DIRECTORY);
    FileOutputFormat.setOutputPath(job, outPath);

    job.setMapperClass(Mapper.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setReducerClass(LLRReducer.class);
    job.setNumReduceTasks(reduceTasks);

    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) 
      throw new IllegalStateException("Job failed!");
  }
}
