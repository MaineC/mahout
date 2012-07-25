package org.apache.mahout.vectorizer;

import org.apache.mahout.vectorizer.collocations.llr.LLRReducer;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;

public class VectorizerJobConfig {
  private Class analyzerClass = DefaultAnalyzer.class;
  private boolean sequentialAccessOutput = false;
  private boolean namedVectors = false;
  private double maxDFSigma = -1.0;
  private boolean processIdf = true;
  private float minLLRValue = LLRReducer.DEFAULT_MIN_LLR;
  private float norm = PartialVectorMerger.NO_NORMALIZING;
  private boolean logNormalize;
  private int reduceTasks = 1;
  private int chunkSize = 100;
  private int minSupport = 2;
  private int maxNGramSize = 1;
  private int maxDFPercent = 99;
  private int minDf = 1;
  
  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result
        + ((analyzerClass == null) ? 0 : analyzerClass.hashCode());
    result = prime * result + chunkSize;
    result = prime * result + (logNormalize ? 1231 : 1237);
    result = prime * result + maxDFPercent;
    long temp;
    temp = Double.doubleToLongBits(maxDFSigma);
    result = prime * result + (int) (temp ^ (temp >>> 32));
    result = prime * result + maxNGramSize;
    result = prime * result + minDf;
    result = prime * result + Float.floatToIntBits(minLLRValue);
    result = prime * result + minSupport;
    result = prime * result + (namedVectors ? 1231 : 1237);
    result = prime * result + Float.floatToIntBits(norm);
    result = prime * result + (processIdf ? 1231 : 1237);
    result = prime * result + reduceTasks;
    result = prime * result + (sequentialAccessOutput ? 1231 : 1237);
    return result;
  }
  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    VectorizerJobConfig other = (VectorizerJobConfig) obj;
    if (analyzerClass == null) {
      if (other.analyzerClass != null)
        return false;
    } else if (!analyzerClass.equals(other.analyzerClass))
      return false;
    if (chunkSize != other.chunkSize)
      return false;
    if (logNormalize != other.logNormalize)
      return false;
    if (maxDFPercent != other.maxDFPercent)
      return false;
    if (Double.doubleToLongBits(maxDFSigma) != Double
        .doubleToLongBits(other.maxDFSigma))
      return false;
    if (maxNGramSize != other.maxNGramSize)
      return false;
    if (minDf != other.minDf)
      return false;
    if (Float.floatToIntBits(minLLRValue) != Float
        .floatToIntBits(other.minLLRValue))
      return false;
    if (minSupport != other.minSupport)
      return false;
    if (namedVectors != other.namedVectors)
      return false;
    if (Float.floatToIntBits(norm) != Float.floatToIntBits(other.norm))
      return false;
    if (processIdf != other.processIdf)
      return false;
    if (reduceTasks != other.reduceTasks)
      return false;
    if (sequentialAccessOutput != other.sequentialAccessOutput)
      return false;
    return true;
  }
  @Override
  public String toString() {
    return "VectorizerConfig [analyzerClass=" + analyzerClass
        + ", sequentialAccessOutput=" + sequentialAccessOutput
        + ", namedVectors=" + namedVectors + ", maxDFSigma=" + maxDFSigma
        + ", processIdf=" + processIdf + ", minLLRValue=" + minLLRValue
        + ", norm=" + norm + ", logNormalize=" + logNormalize
        + ", reduceTasks=" + reduceTasks + ", chunkSize=" + chunkSize
        + ", minSupport=" + minSupport + ", maxNGramSize=" + maxNGramSize
        + ", maxDFPercent=" + maxDFPercent + ", minDf=" + minDf + "]";
  }
  public Class getAnalyzerClass() {
    return analyzerClass;
  }
  public void setAnalyzerClass(Class analyzerClass) {
    this.analyzerClass = analyzerClass;
  }
  public boolean isSequentialAccessOutput() {
    return sequentialAccessOutput;
  }
  public void setSequentialAccessOutput(boolean sequentialAccessOutput) {
    this.sequentialAccessOutput = sequentialAccessOutput;
  }
  public boolean isNamedVectors() {
    return namedVectors;
  }
  public void setNamedVectors(boolean namedVectors) {
    this.namedVectors = namedVectors;
  }
  public double getMaxDFSigma() {
    return maxDFSigma;
  }
  public void setMaxDFSigma(double maxDFSigma) {
    this.maxDFSigma = maxDFSigma;
  }
  public boolean isProcessIdf() {
    return processIdf;
  }
  public void setProcessIdf(boolean processIdf) {
    this.processIdf = processIdf;
  }
  public float getMinLLRValue() {
    return minLLRValue;
  }
  public void setMinLLRValue(float minLLRValue) {
    this.minLLRValue = minLLRValue;
  }
  public float getNorm() {
    return norm;
  }
  public void setNorm(float norm) {
    this.norm = norm;
  }
  public boolean isLogNormalize() {
    return logNormalize;
  }
  public void setLogNormalize(boolean logNormalize) {
    this.logNormalize = logNormalize;
  }
  public int getReduceTasks() {
    return reduceTasks;
  }
  public void setReduceTasks(int reduceTasks) {
    this.reduceTasks = reduceTasks;
  }
  public int getChunkSize() {
    return chunkSize;
  }
  public void setChunkSize(int chunkSize) {
    this.chunkSize = chunkSize;
  }
  public int getMinSupport() {
    return minSupport;
  }
  public void setMinSupport(int minSupport) {
    this.minSupport = minSupport;
  }
  public int getMaxNGramSize() {
    return maxNGramSize;
  }
  public void setMaxNGramSize(int maxNGramSize) {
    this.maxNGramSize = maxNGramSize;
  }
  public int getMaxDFPercent() {
    return maxDFPercent;
  }
  public void setMaxDFPercent(int maxDFPercent) {
    this.maxDFPercent = maxDFPercent;
  }
  public int getMinDf() {
    return minDf;
  }
  public void setMinDf(int minDf) {
    this.minDf = minDf;
  }
}
