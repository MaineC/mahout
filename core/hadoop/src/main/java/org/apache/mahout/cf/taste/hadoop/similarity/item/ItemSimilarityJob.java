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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.IOException;
import java.util.Iterator;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.mahout.cf.taste.common.TopK;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.item.RecommenderConfig;
import org.apache.mahout.cf.taste.hadoop.preparation.PreparePreferenceMatrixJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.mapreduce.AbstractJob;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityConfig;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityJob;
import org.apache.mahout.math.map.OpenIntLongHashMap;

/**
 * <p>Distributed precomputation of the item-item-similarities for Itembased Collaborative Filtering</p>
 *
 * <p>Preferences in the input file should look like {@code userID,itemID[,preferencevalue]}</p>
 *
 * <p>
 * Preference value is optional to accommodate applications that have no notion of a preference value (that is, the user
 * simply expresses a preference for an item, but no degree of preference).
 * </p>
 *
 * <p>
 * The preference value is assumed to be parseable as a {@code double}. The user IDs and item IDs are
 * parsed as {@code long}s.
 * </p>
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>-Dmapred.input.dir=(path): Directory containing one or more text files with the preference data</li>
 * <li>-Dmapred.output.dir=(path): output path where similarity data should be written</li>
 * <li>--similarityClassname (classname): Name of distributed similarity measure class to instantiate or a predefined similarity
 *  from {@link org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasure}</li>
 * <li>--maxSimilaritiesPerItem (integer): Maximum number of similarities considered per item (100)</li>
 * <li>--maxCooccurrencesPerItem (integer): Maximum number of cooccurrences considered per item (100)</li>
 * <li>--booleanData (boolean): Treat input data as having no pref values (false)</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractCLI}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other arguments.</p>
 */
public final class ItemSimilarityJob extends AbstractJob {

  static final String ITEM_ID_INDEX_PATH_STR = ItemSimilarityJob.class.getName() + ".itemIDIndexPathStr";
  static final String MAX_SIMILARITIES_PER_ITEM = ItemSimilarityJob.class.getName() + ".maxSimilarItemsPerItem";

  public static final int DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM = 100;
  public static final int DEFAULT_MAX_PREFS_PER_USER = 1000;
  public static final int DEFAULT_MIN_PREFS_PER_USER = 1;

  public int run(Path inputPath, Path outputPath, RecommenderConfig config) throws IOException, InterruptedException, ClassNotFoundException, InstantiationException, IllegalAccessException {
    return this.run(inputPath, outputPath, config, null, null);
  }

  public int run(Path inputPath, Path outputPath, RecommenderConfig config, Integer start, Integer end) throws IOException, InterruptedException, ClassNotFoundException, InstantiationException, IllegalAccessException {
    Path similarityMatrixPath = getTempPath("similarityMatrix");
    Path prepPath = getTempPath("prepareRatingMatrix");

    AtomicInteger currentPhase = new AtomicInteger();

    if (shouldRunNextPhase(start, end, currentPhase)) {
      PreparePreferenceMatrixJob job = new PreparePreferenceMatrixJob();
      job.setConf(getConf());
      job.run(
          inputPath, prepPath, config.getMaxPrefsPerUser(), config.getMinPrefsPerUser(), 0.0f, config.getBooleanData());
    }

    if (shouldRunNextPhase(start, end, currentPhase)) {
      int numberOfUsers = HadoopUtil.readInt(new Path(prepPath, PreparePreferenceMatrixJob.NUM_USERS),
          getConf());

      RowSimilarityJob job = new RowSimilarityJob();
      job.setConf(getConf());
      RowSimilarityConfig conf = new RowSimilarityConfig();
      conf.setNumberOfCols(numberOfUsers);
      conf.setSimilarityClassName(config.getRowSimilarityConfig().getSimilarityClassName());
      conf.setMaxSimilaritiesPerRow(config.getRowSimilarityConfig().getMaxSimilaritiesPerRow());
      conf.setExcludeSelfSimilarity(true);
      conf.setThreshold(config.getRowSimilarityConfig().getThreshold());
      job.setTempPath(getTempPath());
      job.run(conf, new Path(prepPath, PreparePreferenceMatrixJob.RATING_MATRIX), similarityMatrixPath);
    }

    if (shouldRunNextPhase(start, end, currentPhase)) {
      Job mostSimilarItems = prepareJob(similarityMatrixPath, outputPath, SequenceFileInputFormat.class,
          MostSimilarItemPairsMapper.class, EntityEntityWritable.class, DoubleWritable.class,
          MostSimilarItemPairsReducer.class, EntityEntityWritable.class, DoubleWritable.class, TextOutputFormat.class);
      Configuration mostSimilarItemsConf = mostSimilarItems.getConfiguration();
      mostSimilarItemsConf.set(ITEM_ID_INDEX_PATH_STR,
          new Path(prepPath, PreparePreferenceMatrixJob.ITEMID_INDEX).toString());
      mostSimilarItemsConf.setInt(MAX_SIMILARITIES_PER_ITEM, config.getRowSimilarityConfig().getMaxSimilaritiesPerRow());
      boolean succeeded = mostSimilarItems.waitForCompletion(true);
      if (!succeeded) {
        return -1;
      }
    }

    return 0;
  }

  public static class MostSimilarItemPairsMapper
      extends Mapper<IntWritable,VectorWritable,EntityEntityWritable,DoubleWritable> {

    private OpenIntLongHashMap indexItemIDMap;
    private int maxSimilarItemsPerItem;

    @Override
    protected void setup(Context ctx) {
      Configuration conf = ctx.getConfiguration();
      maxSimilarItemsPerItem = conf.getInt(ItemSimilarityJob.MAX_SIMILARITIES_PER_ITEM, -1);
      indexItemIDMap = TasteHadoopUtils.readItemIDIndexMap(conf.get(ItemSimilarityJob.ITEM_ID_INDEX_PATH_STR), conf);

      Preconditions.checkArgument(maxSimilarItemsPerItem > 0, "maxSimilarItemsPerItem was not correctly set!");
    }

    @Override
    protected void map(IntWritable itemIDIndexWritable, VectorWritable similarityVector, Context ctx)
      throws IOException, InterruptedException {

      int itemIDIndex = itemIDIndexWritable.get();

      TopK<SimilarItem> topKMostSimilarItems =
          new TopK<SimilarItem>(maxSimilarItemsPerItem, SimilarItem.COMPARE_BY_SIMILARITY);

      Iterator<Vector.Element> similarityVectorIterator = similarityVector.get().iterateNonZero();

      while (similarityVectorIterator.hasNext()) {
        Vector.Element element = similarityVectorIterator.next();
        topKMostSimilarItems.offer(new SimilarItem(indexItemIDMap.get(element.index()), element.get()));
      }

      long itemID = indexItemIDMap.get(itemIDIndex);
      for (SimilarItem similarItem : topKMostSimilarItems.retrieve()) {
        long otherItemID = similarItem.getItemID();
        if (itemID < otherItemID) {
          ctx.write(new EntityEntityWritable(itemID, otherItemID), new DoubleWritable(similarItem.getSimilarity()));
        } else {
          ctx.write(new EntityEntityWritable(otherItemID, itemID), new DoubleWritable(similarItem.getSimilarity()));
        }
      }
    }
  }

  static class MostSimilarItemPairsReducer
      extends Reducer<EntityEntityWritable,DoubleWritable,EntityEntityWritable,DoubleWritable> {
    @Override
    protected void reduce(EntityEntityWritable pair, Iterable<DoubleWritable> values, Context ctx)
        throws IOException, InterruptedException {
      ctx.write(pair, values.iterator().next());
    }
  }
}
