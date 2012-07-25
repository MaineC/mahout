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

package org.apache.mahout.classifier.naivebayes.training;

import com.google.common.base.Splitter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.classifier.naivebayes.BayesUtils;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.mapreduce.AbstractJob;
import org.apache.mahout.common.mapreduce.VectorSumReducer;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import static org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesConfig.*;

/**
 * This class trains a Naive Bayes Classifier (Parameters for both Naive Bayes and Complementary Naive Bayes)
 */
public final class TrainNaiveBayesJob extends AbstractJob {

  public int run(Path input, Path output, Path labPath, boolean trainComplementary) throws IOException, InterruptedException, ClassNotFoundException {
    return this.run(input, output, labPath, 1.0f, trainComplementary);
  }
  public int run(Path input, Path output, boolean trainComplementary, String labels) throws IOException, InterruptedException, ClassNotFoundException {
    return this.run(input, output, 1.0f, trainComplementary, labels);
  }
  public int run(Path input, Path output, Path labPath, float alphaI, boolean trainComplementary) throws IOException, InterruptedException, ClassNotFoundException {
    return this.run(input, output, labPath, alphaI, trainComplementary, null);
  }
  public int run(Path input, Path output, float alphaI, boolean trainComplementary, String labels) throws IOException, InterruptedException, ClassNotFoundException {
    return this.run(input, output, null, alphaI, trainComplementary, labels);
  }
  public int run(Path input, Path output, Path labPath, float alphaI, boolean trainComplementary, String labels) throws IOException, InterruptedException, ClassNotFoundException {
    if (labPath == null) {
      labPath = getTempPath("labelIndex");
    }
    long labelSize = createLabelIndex(labPath, labels, input);

    HadoopUtil.setSerializations(getConf());
    HadoopUtil.cacheFiles(labPath, getConf());

    //add up all the vectors with the same labels, while mapping the labels into our index
    Job indexInstances = prepareJob(input, getTempPath(SUMMED_OBSERVATIONS), SequenceFileInputFormat.class,
            IndexInstancesMapper.class, IntWritable.class, VectorWritable.class, VectorSumReducer.class, IntWritable.class,
            VectorWritable.class, SequenceFileOutputFormat.class);
    indexInstances.setCombinerClass(VectorSumReducer.class);
    boolean succeeded = indexInstances.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }
    //sum up all the weights from the previous step, per label and per feature
    Job weightSummer = prepareJob(getTempPath(SUMMED_OBSERVATIONS), getTempPath(WEIGHTS),
            SequenceFileInputFormat.class, WeightsMapper.class, Text.class, VectorWritable.class, VectorSumReducer.class,
            Text.class, VectorWritable.class, SequenceFileOutputFormat.class);
    weightSummer.getConfiguration().set(WeightsMapper.NUM_LABELS, String.valueOf(labelSize));
    weightSummer.setCombinerClass(VectorSumReducer.class);
    succeeded = weightSummer.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }
    //put the per label and per feature vectors into the cache
    HadoopUtil.cacheFiles(getTempPath(WEIGHTS), getConf());
    //calculate the Thetas, write out to LABEL_THETA_NORMALIZER vectors -- TODO: add reference here to the part of the Rennie paper that discusses this
    Job thetaSummer = prepareJob(getTempPath(SUMMED_OBSERVATIONS), getTempPath(THETAS),
            SequenceFileInputFormat.class, ThetaMapper.class, Text.class, VectorWritable.class, VectorSumReducer.class,
            Text.class, VectorWritable.class, SequenceFileOutputFormat.class);
    thetaSummer.setCombinerClass(VectorSumReducer.class);
    thetaSummer.getConfiguration().setFloat(ThetaMapper.ALPHA_I, alphaI);
    thetaSummer.getConfiguration().setBoolean(ThetaMapper.TRAIN_COMPLEMENTARY, trainComplementary);
    succeeded = thetaSummer.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }
    //validate our model and then write it out to the official output
    NaiveBayesModel naiveBayesModel = BayesUtils.readModelFromDir(getTempPath(), getConf());
    naiveBayesModel.validate();
    naiveBayesModel.serialize(output, getConf());

    return 0;
  }

  private long createLabelIndex(Path labPath, String labels, Path input) throws IOException {
    long labelSize = 0;
    if (labels != null) {
      Iterable<String> labelIter = Splitter.on(",").split(labels);
      labelSize = BayesUtils.writeLabelIndex(getConf(), labelIter, labPath);
    } else {
      SequenceFileDirIterable<Text, IntWritable> iterable =
              new SequenceFileDirIterable<Text, IntWritable>(input, PathType.LIST, PathFilters.logsCRCFilter(), getConf());
      labelSize = BayesUtils.writeLabelIndex(getConf(), labPath, iterable);
    }
    return labelSize;
  }

}
