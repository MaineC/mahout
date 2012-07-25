package org.apache.mahout.cf.taste.hadoop.item;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityConfig;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityJob;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasures;

public class RecommenderCLI extends AbstractCLI {

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("numRecommendations", "n", "Number of recommendations per user",
            String.valueOf(AggregateAndRecommendReducer.DEFAULT_NUM_RECOMMENDATIONS));
    addOption("usersFile", null, "File of users to recommend for", null);
    addOption("itemsFile", null, "File of items to recommend for", null);
    addOption("filterFile", "f", "File containing comma-separated userID,itemID pairs. Used to exclude the item from "
            + "the recommendations for that user (optional)", null);
    addOption("booleanData", "b", "Treat input as without pref values", Boolean.FALSE.toString());
    addOption("maxPrefsPerUser", "mxp",
            "Maximum number of preferences considered per user in final recommendation phase",
            String.valueOf(UserVectorSplitterMapper.DEFAULT_MAX_PREFS_PER_USER_CONSIDERED));
    addOption("minPrefsPerUser", "mp", "ignore users with less preferences than this in the similarity computation "
            + "(default: " + RecommenderConfig.DEFAULT_MIN_PREFS_PER_USER + ')', String.valueOf(RecommenderConfig.DEFAULT_MIN_PREFS_PER_USER));
    addOption("maxSimilaritiesPerItem", "m", "Maximum number of similarities considered per item ",
            String.valueOf(RecommenderConfig.DEFAULT_MAX_SIMILARITIES_PER_ITEM));
    addOption("maxPrefsPerUserInItemSimilarity", "mppuiis", "max number of preferences to consider per user in the " 
            + "item similarity computation phase, users with more preferences will be sampled down (default: " +
            RecommenderConfig.DEFAULT_MAX_PREFS_PER_USER + ')', String.valueOf(RecommenderConfig.DEFAULT_MAX_PREFS_PER_USER));
    addOption("similarityClassname", "s", "Name of distributed similarity measures class to instantiate, " 
            + "alternatively use one of the predefined similarities (" + VectorSimilarityMeasures.list() + ')', true);
    addOption("threshold", "tr", "discard item pairs with a similarity value below this", false);

    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    RecommenderConfig config = new RecommenderConfig();
    config.setNumRecommendations(Integer.parseInt(getOption("numRecommendations")));
    config.setUsersFile(getOption("usersFile"));
    config.setItemsFile(getOption("itemsFile"));
    config.setFilterFile(getOption("filterFile"));
    config.setBooleanData(Boolean.valueOf(getOption("booleanData")));
    config.setMaxPrefsPerUser(Integer.parseInt(getOption("maxPrefsPerUser")));
    config.setMinPrefsPerUser(Integer.parseInt(getOption("minPrefsPerUser")));
    config.setMaxPrefsPerUserInItemSimilarity(Integer.parseInt(getOption("maxPrefsPerUserInItemSimilarity")));
    config.setMaxSimilaritiesPerItem(Integer.parseInt(getOption("maxSimilaritiesPerItem")));
    config.getRowSimilarityConfig().setSimilarityClassName(getOption("similarityClassname"));
    double threshold = hasOption("threshold") ?
                 Double.parseDouble(getOption("threshold")) : RowSimilarityJob.NO_THRESHOLD;
    RecommenderJob job = new RecommenderJob();
    RowSimilarityConfig rsc = config.getRowSimilarityConfig();
    rsc.setDefaults();
    rsc.setThreshold(threshold);
    Integer start = null;
    Integer end = null;
    String sp = getOption(parsedArgs, "--startPhase");
    String ep = getOption(parsedArgs, "--endPhase");
    if (sp != null) start = new Integer(sp);
    if (ep != null) end = new Integer(ep);
    return job.run(config,  getInputPath(), getOutputPath(), start, end);
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new RecommenderCLI(), args);
  }

}
