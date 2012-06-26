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

package org.apache.mahout.math.hadoop.similarity;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

import static org.apache.mahout.math.hadoop.similarity.VectorDistanceConfig.*;

/**
 * This class does a Map-side join between seed vectors (the map side can also be a Cluster) and a list of other vectors
 * and emits the a tuple of seed id, other id, distance.  It is a more generic version of KMean's mapper
 */
public class VectorDistanceSimilarityJob extends Configured {
  public void run(Configuration conf,
      Path input,
      Path seeds,
      Path output,
      DistanceMeasure measure) throws IOException, ClassNotFoundException, InterruptedException {
    this.run(conf, input, seeds, output, measure, "pw");
  }

  public void run(Configuration conf,
                         Path input,
                         Path seeds,
                         Path output,
                         DistanceMeasure measure, String outType)
    throws IOException, ClassNotFoundException, InterruptedException {
    conf.set(DISTANCE_MEASURE_KEY, measure.getClass().getName());
    conf.set(SEEDS_PATH_KEY, seeds.toString());
    Job job = new Job(conf, "Vector Distance Similarity: seeds: " + seeds + " input: " + input);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    if ("pw".equalsIgnoreCase(outType)) {
      job.setMapOutputKeyClass(StringTuple.class);
      job.setOutputKeyClass(StringTuple.class);
      job.setMapOutputValueClass(DoubleWritable.class);
      job.setOutputValueClass(DoubleWritable.class);
      job.setMapperClass(VectorDistanceMapper.class);
    } else if ("v".equalsIgnoreCase(outType)) {
      job.setMapOutputKeyClass(Text.class);
      job.setOutputKeyClass(Text.class);
      job.setMapOutputValueClass(VectorWritable.class);
      job.setOutputValueClass(VectorWritable.class);
      job.setMapperClass(VectorDistanceInvertedMapper.class);
    } else {
      throw new IllegalArgumentException("Invalid outType specified: " + outType);
    }


    job.setNumReduceTasks(0);
    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.setJarByClass(VectorDistanceSimilarityJob.class);
    HadoopUtil.delete(conf, output);
    if (!job.waitForCompletion(true)) {
      throw new IllegalStateException("VectorDistance Similarity failed processing " + seeds);
    }
  }
}
