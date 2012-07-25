package org.apache.mahout.cf.taste.hadoop.item;

import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityConfig;

public class RecommenderConfig {
  public static final int DEFAULT_MAX_SIMILARITIES_PER_ITEM = 100;
  public static final int DEFAULT_MAX_PREFS_PER_USER = 1000;
  public static final int DEFAULT_MIN_PREFS_PER_USER = 1;
  public static final String BOOLEAN_DATA = "BOOLEAN_DATA";


  private RowSimilarityConfig config = new RowSimilarityConfig();
  private int maxPrefsPerUser = DEFAULT_MAX_PREFS_PER_USER;
  private String usersFile;
  private String itemsFile;
  private String filterFile;
  private boolean booleanData;
  private int maxPrefsPerUserInItemSimilarity = DEFAULT_MAX_SIMILARITIES_PER_ITEM;
  private int minPrefsPerUser = DEFAULT_MIN_PREFS_PER_USER;
  private int numRecommendations = 2;

  
  public RowSimilarityConfig getRowSimilarityConfig() {
    return this.config;
  }

  public int getMaxPrefsPerUser() {
    return maxPrefsPerUser;
  }

  public String getUsersFile() {
    return usersFile;
  }

  public boolean getBooleanData() {
    return booleanData;
  }

  public Integer getMaxPrefsPerUserInItemSimilarity() {
    return maxPrefsPerUserInItemSimilarity;
  }

  public int getMinPrefsPerUser() {
    return minPrefsPerUser;
  }

  public String getFilterFile() {
    return filterFile;
  }

  public String getItemsFile() {
    return itemsFile;
  }

  public int getNumRecommendations() {
    return numRecommendations;
  }
  
  public void setNumRecommendations(int num) {
    this.numRecommendations = num;
  }

  public void setUsersFile(String option) {
    this.usersFile = option;
  }

  public void setItemsFile(String option) {
    this.itemsFile = option;
  }

  public void setFilterFile(String option) {
    this.filterFile = option;
  }

  public void setBooleanData(boolean option) {
    this.booleanData = option;
  }

  public void setMaxPrefsPerUser(int opt) {
    this.maxPrefsPerUser = opt;
  }

  public void setMinPrefsPerUser(int opt) {
    this.minPrefsPerUser = opt;
  }

  public void setMaxPrefsPerUserInItemSimilarity(int opt) {
    this.maxPrefsPerUserInItemSimilarity = opt;
  }

  public void setMaxSimilaritiesPerItem(int opt) {
    this.config.setMaxSimilaritiesPerRow(opt);
  }
}
