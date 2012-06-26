/*
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

package org.apache.mahout.cf.taste.hadoop.item;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.preparation.PreparePreferenceMatrixJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.mapreduce.AbstractJob;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityConfig;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityJob;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * <p>Runs a completely distributed recommender job as a series of mapreduces.</p>
 * <p/>
 * <p>Preferences in the input file should look like {@code userID, itemID[, preferencevalue]}</p>
 * <p/>
 * <p>
 * Preference value is optional to accommodate applications that have no notion of a preference value (that is, the user
 * simply expresses a preference for an item, but no degree of preference).
 * </p>
 * <p/>
 * <p>
 * The preference value is assumed to be parseable as a {@code double}. The user IDs and item IDs are
 * parsed as {@code long}s.
 * </p>
 * <p/>
 * <p>Command line arguments specific to this class are:</p>
 * <p/>
 * <ol>
 * <li>--input(path): Directory containing one or more text files with the preference data</li>
 * <li>--output(path): output path where recommender output should go</li>
 * <li>--similarityClassname (classname): Name of vector similarity class to instantiate or a predefined similarity
 * from {@link org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasure}</li>
 * <li>--usersFile (path): only compute recommendations for user IDs contained in this file (optional)</li>
 * <li>--itemsFile (path): only include item IDs from this file in the recommendations (optional)</li>
 * <li>--filterFile (path): file containing comma-separated userID,itemID pairs. Used to exclude the item from the
 * recommendations for that user (optional)</li>
 * <li>--numRecommendations (integer): Number of recommendations to compute per user (10)</li>
 * <li>--booleanData (boolean): Treat input data as having no pref values (false)</li>
 * <li>--maxPrefsPerUser (integer): Maximum number of preferences considered per user in  final recommendation phase (10)</li>
 * <li>--maxSimilaritiesPerItem (integer): Maximum number of similarities considered per item (100)</li>
 * <li>--minPrefsPerUser (integer): ignore users with less preferences than this in the similarity computation (1)</li>
 * <li>--maxPrefsPerUserInItemSimilarity (integer): max number of preferences to consider per user in the item similarity computation phase,
 * users with more preferences will be sampled down (1000)</li>
 * <li>--threshold (double): discard item pairs with a similarity value below this</li>
 * </ol>
 * <p/>
 * <p>General command line options are documented in {@link AbstractCLI}.</p>
 * <p/>
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other
 * arguments.</p>
 */
public final class RecommenderJob extends AbstractJob {
  public int run(RecommenderConfig config) throws IOException, InterruptedException, ClassNotFoundException, InstantiationException, IllegalAccessException {
    return this.run(config, new Path(getConf().get("mapred.input.dir")), new Path(getConf().get("mapred.output.dir")), null, null);
  }

  public int run(RecommenderConfig config, Path input, Path output, Integer start, Integer end) throws IOException, InterruptedException, ClassNotFoundException, InstantiationException, IllegalAccessException {
    Path prepPath = getTempPath("preparePreferenceMatrix");
    Path similarityMatrixPath = getTempPath("similarityMatrix");
    Path prePartialMultiplyPath1 = getTempPath("prePartialMultiply1");
    Path prePartialMultiplyPath2 = getTempPath("prePartialMultiply2");
    Path explicitFilterPath = getTempPath("explicitFilterPath");
    Path partialMultiplyPath = getTempPath("partialMultiply");

    AtomicInteger currentPhase = new AtomicInteger();

    int numberOfUsers = -1;

    if (shouldRunNextPhase(start, end, currentPhase)) {
      PreparePreferenceMatrixJob job = new PreparePreferenceMatrixJob();
      job.setConf(getConf());
      job.run(input, prepPath, config.getMaxPrefsPerUserInItemSimilarity(), config.getMinPrefsPerUser(), 0.0f, config.getBooleanData());
      numberOfUsers = HadoopUtil.readInt(new Path(prepPath, PreparePreferenceMatrixJob.NUM_USERS), getConf());
    }


    if (shouldRunNextPhase(start, end, currentPhase)) {

      /* special behavior if phase 1 is skipped */
      if (numberOfUsers == -1) {
        numberOfUsers = (int) HadoopUtil.countRecords(new Path(prepPath, PreparePreferenceMatrixJob.USER_VECTORS),
                PathType.LIST, null, getConf());
      }

      /* Once DistributedRowMatrix uses the hadoop 0.20 API, we should refactor this call to something like
       * new DistributedRowMatrix(...).rowSimilarity(...) */
      //calculate the co-occurrence matrix
      RowSimilarityJob job = new RowSimilarityJob();
      job.setTempPath(getTempPath());
      job.setConf(getConf());
      RowSimilarityConfig rowSimConf = config.getRowSimilarityConfig();
      rowSimConf.setNumberOfCols(numberOfUsers);
      job.run(rowSimConf, new Path(prepPath, PreparePreferenceMatrixJob.RATING_MATRIX), similarityMatrixPath);
    }

    //start the multiplication of the co-occurrence matrix by the user vectors
    if (shouldRunNextPhase(start, end, currentPhase)) {
      Job prePartialMultiply1 = prepareJob(
              similarityMatrixPath, prePartialMultiplyPath1, SequenceFileInputFormat.class,
              SimilarityMatrixRowWrapperMapper.class, VarIntWritable.class, VectorOrPrefWritable.class,
              Reducer.class, VarIntWritable.class, VectorOrPrefWritable.class,
              SequenceFileOutputFormat.class);
      boolean succeeded = prePartialMultiply1.waitForCompletion(true);
      if (!succeeded) 
        return -1;
      //continue the multiplication
      Job prePartialMultiply2 = prepareJob(new Path(prepPath, PreparePreferenceMatrixJob.USER_VECTORS),
              prePartialMultiplyPath2, SequenceFileInputFormat.class, UserVectorSplitterMapper.class, VarIntWritable.class,
              VectorOrPrefWritable.class, Reducer.class, VarIntWritable.class, VectorOrPrefWritable.class,
              SequenceFileOutputFormat.class);
      if (config.getUsersFile() != null) {
        prePartialMultiply2.getConfiguration().set(UserVectorSplitterMapper.USERS_FILE, config.getUsersFile());
      }
      prePartialMultiply2.getConfiguration().setInt(UserVectorSplitterMapper.MAX_PREFS_PER_USER_CONSIDERED,
              config.getMaxPrefsPerUser());
      succeeded = prePartialMultiply2.waitForCompletion(true);
      if (!succeeded) 
        return -1;
      //finish the job
      Job partialMultiply = prepareJob(
              new Path(prePartialMultiplyPath1 + "," + prePartialMultiplyPath2), partialMultiplyPath,
              SequenceFileInputFormat.class, Mapper.class, VarIntWritable.class, VectorOrPrefWritable.class,
              ToVectorAndPrefReducer.class, VarIntWritable.class, VectorAndPrefsWritable.class,
              SequenceFileOutputFormat.class);
      setS3SafeCombinedInputPath(partialMultiply, getTempPath(), prePartialMultiplyPath1, prePartialMultiplyPath2);
      succeeded = partialMultiply.waitForCompletion(true);
      if (!succeeded) 
        return -1;
    }

    if (shouldRunNextPhase(start, end, currentPhase)) {
      //filter out any users we don't care about
      /* convert the user/item pairs to filter if a filterfile has been specified */
      if (config.getFilterFile() != null) {
        Job itemFiltering = prepareJob(new Path(config.getFilterFile()), explicitFilterPath, TextInputFormat.class,
                ItemFilterMapper.class, VarLongWritable.class, VarLongWritable.class,
                ItemFilterAsVectorAndPrefsReducer.class, VarIntWritable.class, VectorAndPrefsWritable.class,
                SequenceFileOutputFormat.class);
        boolean succeeded = itemFiltering.waitForCompletion(true);
        if (!succeeded) 
          return -1;
      }

      String aggregateAndRecommendInput = partialMultiplyPath.toString();
      if (config.getFilterFile() != null) {
        aggregateAndRecommendInput += "," + explicitFilterPath;
      }
      //extract out the recommendations
      Job aggregateAndRecommend = prepareJob(
              new Path(aggregateAndRecommendInput), output, SequenceFileInputFormat.class,
              PartialMultiplyMapper.class, VarLongWritable.class, PrefAndSimilarityColumnWritable.class,
              AggregateAndRecommendReducer.class, VarLongWritable.class, RecommendedItemsWritable.class,
              TextOutputFormat.class);
      Configuration aggregateAndRecommendConf = aggregateAndRecommend.getConfiguration();
      if (config.getItemsFile() != null) {
        aggregateAndRecommendConf.set(AggregateAndRecommendReducer.ITEMS_FILE, config.getItemsFile());
      }

      if (config.getFilterFile() != null) {
        setS3SafeCombinedInputPath(aggregateAndRecommend, getTempPath(), partialMultiplyPath, explicitFilterPath);
      }
      setIOSort(aggregateAndRecommend);
      aggregateAndRecommendConf.set(AggregateAndRecommendReducer.ITEMID_INDEX_PATH,
              new Path(prepPath, PreparePreferenceMatrixJob.ITEMID_INDEX).toString());
      aggregateAndRecommendConf.setInt(AggregateAndRecommendReducer.NUM_RECOMMENDATIONS, config.getNumRecommendations());
      aggregateAndRecommendConf.setBoolean(RecommenderConfig.BOOLEAN_DATA, config.getBooleanData());
      boolean succeeded = aggregateAndRecommend.waitForCompletion(true);
      if (!succeeded) 
        return -1;
    }

    return 0;
  }

  private static void setIOSort(JobContext job) {
    Configuration conf = job.getConfiguration();
    conf.setInt("io.sort.factor", 100);
    String javaOpts = conf.get("mapred.map.child.java.opts"); // new arg name
    if (javaOpts == null) {
      javaOpts = conf.get("mapred.child.java.opts"); // old arg name
    }
    int assumedHeapSize = 512;
    if (javaOpts != null) {
      Matcher m = Pattern.compile("-Xmx([0-9]+)([mMgG])").matcher(javaOpts);
      if (m.find()) {
        assumedHeapSize = Integer.parseInt(m.group(1));
        String megabyteOrGigabyte = m.group(2);
        if ("g".equalsIgnoreCase(megabyteOrGigabyte)) {
          assumedHeapSize *= 1024;
        }
      }
    }
    // Cap this at 1024MB now; see https://issues.apache.org/jira/browse/MAPREDUCE-2308
    conf.setInt("io.sort.mb", Math.min(assumedHeapSize / 2, 1024));
    // For some reason the Merger doesn't report status for a long time; increase
    // timeout when running these jobs
    conf.setInt("mapred.task.timeout", 60 * 60 * 1000);
  }

}
