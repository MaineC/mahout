package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.item.RecommenderConfig;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityJob;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasures;

public class ItemSimilarityCLI extends AbstractCLI {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ItemSimilarityCLI(), args);
  }
  
  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("similarityClassname", "s", "Name of distributed similarity measures class to instantiate, " 
        + "alternatively use one of the predefined similarities (" + VectorSimilarityMeasures.list() + ')');
    addOption("maxSimilaritiesPerItem", "m", "try to cap the number of similar items per item to this number "
        + "(default: " + ItemSimilarityJob.DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM + ')',
        String.valueOf(ItemSimilarityJob.DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM));
    addOption("maxPrefsPerUser", "mppu", "max number of preferences to consider per user, " 
        + "users with more preferences will be sampled down (default: " + ItemSimilarityJob.DEFAULT_MAX_PREFS_PER_USER + ')',
        String.valueOf(ItemSimilarityJob.DEFAULT_MAX_PREFS_PER_USER));
    addOption("minPrefsPerUser", "mp", "ignore users with less preferences than this "
        + "(default: " + ItemSimilarityJob.DEFAULT_MIN_PREFS_PER_USER + ')', String.valueOf(ItemSimilarityJob.DEFAULT_MIN_PREFS_PER_USER));
    addOption("booleanData", "b", "Treat input as without pref values", String.valueOf(Boolean.FALSE));
    addOption("threshold", "tr", "discard item pairs with a similarity value below this", false);

    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    String similarityClassName = getOption("similarityClassname");
    int maxSimilarItemsPerItem = Integer.parseInt(getOption("maxSimilaritiesPerItem"));
    int maxPrefsPerUser = Integer.parseInt(getOption("maxPrefsPerUser"));
    int minPrefsPerUser = Integer.parseInt(getOption("minPrefsPerUser"));
    boolean booleanData = Boolean.valueOf(getOption("booleanData"));

    double threshold = hasOption("threshold") ?
        Double.parseDouble(getOption("threshold")) : RowSimilarityJob.NO_THRESHOLD;

    RecommenderConfig config = new RecommenderConfig();
    config.setBooleanData(booleanData);
    config.setMaxPrefsPerUser(maxPrefsPerUser);
    config.setMinPrefsPerUser(minPrefsPerUser);
    config.setMaxSimilaritiesPerItem(maxSimilarItemsPerItem);
    config.getRowSimilarityConfig().setDefaults();
    config.getRowSimilarityConfig().setSimilarityClassName(similarityClassName);
    config.getRowSimilarityConfig().setThreshold(threshold);
    
    String startOpt = getOption("--startPhase");
    int start = 0;
    if (startOpt != null) {
      start = Integer.parseInt(startOpt);
    }
    String endOpt = getOption("--endPhase");
    int end = 0;
    if (endOpt != null) {
      end = Integer.parseInt(endOpt);
    }
    
    ItemSimilarityJob job = new ItemSimilarityJob();
    return job.run(getInputPath(), getOutputPath(), config, start, end);
  }
}
