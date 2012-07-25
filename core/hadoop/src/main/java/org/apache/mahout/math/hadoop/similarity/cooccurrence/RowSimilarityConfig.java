package org.apache.mahout.math.hadoop.similarity.cooccurrence;

public class RowSimilarityConfig {
  public static final int DEFAULT_MAX_SIMILARITIES_PER_ROW = 100;

  private Integer numberOfCols;
  private String similarityClassName;
  private int maxSimilaritiesPerRow = DEFAULT_MAX_SIMILARITIES_PER_ROW;
  private boolean excludeSelfSimilarity = false;
  private double threshold = RowSimilarityJob.NO_THRESHOLD;

  public Integer getNumberOfCols() {
    return numberOfCols;
  }
  public RowSimilarityConfig setNumberOfCols(int numberOfCols) {
    this.numberOfCols = numberOfCols;
    return this;
  }
  public String getSimilarityClassName() {
    return similarityClassName;
  }
  public RowSimilarityConfig setSimilarityClassName(String similarityClassName) {
    this.similarityClassName = similarityClassName;
    return this;
  }
  public int getMaxSimilaritiesPerRow() {
    return maxSimilaritiesPerRow;
  }
  public RowSimilarityConfig setMaxSimilaritiesPerRow(int maxSimilaritiesPerRow) {
    this.maxSimilaritiesPerRow = maxSimilaritiesPerRow;
    return this;
  }
  public boolean getExcludeSelfSimilarity() {
    return excludeSelfSimilarity;
  }
  public RowSimilarityConfig setExcludeSelfSimilarity(boolean excludeSelfSimilarity) {
    this.excludeSelfSimilarity = excludeSelfSimilarity;
    return this;
  }
  public double getThreshold() {
    return threshold;
  }
  public RowSimilarityConfig setThreshold(double threshold) {
    this.threshold = threshold;
    return this;
  }

  public RowSimilarityConfig setDefaults() {
    return this;
  }
}
