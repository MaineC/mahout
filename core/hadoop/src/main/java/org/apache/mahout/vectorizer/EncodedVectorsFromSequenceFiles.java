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

import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.mapreduce.AbstractJob;
import org.apache.mahout.vectorizer.encoders.LuceneTextValueEncoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Converts a given set of sequence files into SparseVectors
 */
public final class EncodedVectorsFromSequenceFiles extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(EncodedVectorsFromSequenceFiles.class);

  public int run(Path input, Path output, String analyzerClass, String encoderClass, String encoderName, boolean sequentialAccessOutput,
      boolean namedVectors, int cardinality) throws IOException, ClassNotFoundException, InterruptedException {
  SimpleTextEncodingVectorizer vectorizer = new SimpleTextEncodingVectorizer();
    VectorizerConfig config = new VectorizerConfig(getConf(), analyzerClass, encoderClass, encoderName, sequentialAccessOutput,
            namedVectors, cardinality);

    vectorizer.createVectors(input, output, config);

    return 0;
  }

  public void run(Path inputPath, Path outputPath, boolean sequential,
      boolean named) throws IOException, ClassNotFoundException, InterruptedException {
    this.run(inputPath, outputPath, DefaultAnalyzer.class.getName(), LuceneTextValueEncoder.class.getName(),
        "text", sequential, named, 5000);
    
  }

}
