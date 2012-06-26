package org.apache.mahout.cf.taste.hadoop.preparation;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;

public class PreparePreferenceMatrixCLI extends AbstractCLI {
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new PreparePreferenceMatrixCLI(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("maxPrefsPerUser", "mppu", "max number of preferences to consider per user, " 
            + "users with more preferences will be sampled down");
    addOption("minPrefsPerUser", "mp", "ignore users with less preferences than this "
            + "(default: " + PreparePreferenceMatrixJob.DEFAULT_MIN_PREFS_PER_USER + ')', String.valueOf(PreparePreferenceMatrixJob.DEFAULT_MIN_PREFS_PER_USER));
    addOption("booleanData", "b", "Treat input as without pref values", Boolean.FALSE.toString());
    addOption("ratingShift", "rs", "shift ratings by this value", "0.0");

    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    int minPrefsPerUser = Integer.parseInt(getOption("minPrefsPerUser"));
    boolean booleanData = Boolean.valueOf(getOption("booleanData"));
    float ratingShift = Float.parseFloat(getOption("ratingShift"));
    Integer samplingSize = null;
    if (hasOption("maxPrefsPerUser")) {
      samplingSize = Integer.parseInt(getOption("maxPrefsPerUser"));
    }

    PreparePreferenceMatrixJob job = new PreparePreferenceMatrixJob();
    return job.run(getInputPath(), getOutputPath(), samplingSize, minPrefsPerUser, ratingShift, booleanData);
  }
}
