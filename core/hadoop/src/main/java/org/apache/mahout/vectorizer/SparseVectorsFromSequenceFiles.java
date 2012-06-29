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

package org.apache.mahout.vectorizer;

import java.io.IOException;
import java.util.List;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.mapreduce.AbstractJob;
import org.apache.mahout.math.hadoop.stats.BasicStats;
import org.apache.mahout.vectorizer.collocations.llr.LLRReducer;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.term.DictionaryConfig;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Converts a given set of sequence files into SparseVectors
 */
public final class SparseVectorsFromSequenceFiles extends AbstractJob {
  
  private static final Logger log = LoggerFactory.getLogger(SparseVectorsFromSequenceFiles.class);

  public int run(Path inputDir, Path outputDir, VectorizerJobConfig config) throws IOException, InterruptedException, ClassNotFoundException {
      Path tokenizedPath = new Path(outputDir, DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
      //TODO: move this into DictionaryVectorizer , and then fold SparseVectorsFrom with EncodedVectorsFrom to have one framework for all of this.
      DocumentProcessor.tokenizeDocuments(inputDir, config.getAnalyzerClass(), tokenizedPath, getConf());

      boolean shouldPrune = config.getMaxDFSigma() >=0.0;
      String tfDirName = shouldPrune ? DictionaryConfig.DOCUMENT_VECTOR_OUTPUT_FOLDER+"-toprune" : DictionaryConfig.DOCUMENT_VECTOR_OUTPUT_FOLDER;

      if (!config.isProcessIdf()) {
        DictionaryVectorizer.createTermFrequencyVectors(tokenizedPath, outputDir, tfDirName, getConf(), config.getMinSupport(), config.getMaxNGramSize(),
          config.getMinLLRValue(), config.getNorm(), config.isLogNormalize(), config.getReduceTasks(), config.getChunkSize(), config.isSequentialAccessOutput(), config.isNamedVectors());
      } else {
        DictionaryVectorizer.createTermFrequencyVectors(tokenizedPath, outputDir, tfDirName, getConf(), config.getMinSupport(), config.getMaxNGramSize(),
          config.getMinLLRValue(), -1.0f, false, config.getReduceTasks(), config.getChunkSize(), config.isSequentialAccessOutput(), config.isNamedVectors());
      }
      Pair<Long[], List<Path>> docFrequenciesFeatures = null;
       // Should document frequency features be processed
       if (shouldPrune || config.isProcessIdf()) {
         docFrequenciesFeatures = TFIDFConverter.calculateDF(new Path(outputDir, tfDirName),
                 outputDir, getConf(), config.getChunkSize());
       }

       long maxDF = config.getMaxDFPercent(); //if we are pruning by std dev, then this will get changed
       if (shouldPrune) {
         Path dfDir = new Path(outputDir, TFIDFConverter.WORDCOUNT_OUTPUT_FOLDER);
         Path stdCalcDir = new Path(outputDir, HighDFWordsPruner.STD_CALC_DIR);

         // Calculate the standard deviation
         double stdDev = BasicStats.stdDevForGivenMean(dfDir, stdCalcDir, 0.0, getConf());
         long vectorCount = docFrequenciesFeatures.getFirst()[1];
         maxDF = (int) (100.0 * config.getMaxDFSigma() * stdDev / vectorCount);

         // Prune the term frequency vectors
         Path tfDir = new Path(outputDir, tfDirName);
         Path prunedTFDir = new Path(outputDir, DictionaryConfig.DOCUMENT_VECTOR_OUTPUT_FOLDER);
         Path prunedPartialTFDir = new Path(outputDir, DictionaryConfig.DOCUMENT_VECTOR_OUTPUT_FOLDER
                 + "-partial");
         if (config.isProcessIdf()) {
           HighDFWordsPruner.pruneVectors(tfDir,
                                          prunedTFDir,
                                          prunedPartialTFDir,
                                          maxDF,
                                          getConf(),
                                          docFrequenciesFeatures,
                                          -1.0f,
                                          false,
                                          config.getReduceTasks());
         } else {
           HighDFWordsPruner.pruneVectors(tfDir,
                                          prunedTFDir,
                                          prunedPartialTFDir,
                                          maxDF,
                                          getConf(),
                                          docFrequenciesFeatures,
                                          config.getNorm(),
                                          config.isLogNormalize(),
                                          config.getReduceTasks());
         }
         HadoopUtil.delete(new Configuration(getConf()), tfDir);
       }
      if (config.isProcessIdf()) {
          TFIDFConverter.processTfIdf(
                 new Path(outputDir, DictionaryConfig.DOCUMENT_VECTOR_OUTPUT_FOLDER),
                 outputDir, getConf(), docFrequenciesFeatures, config.getMinDf(), maxDF, config.getNorm(), config.isLogNormalize(),
                 config.isSequentialAccessOutput(), config.isNamedVectors(), config.getReduceTasks());
      }
    return 0;
  }
  
}
