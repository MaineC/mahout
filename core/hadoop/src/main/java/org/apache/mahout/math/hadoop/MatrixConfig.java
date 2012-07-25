package org.apache.mahout.math.hadoop;

import org.apache.hadoop.fs.Path;

public class MatrixConfig {

  Path inputPath;
  int rows, cols;
  public Path getInputPath() {
    return inputPath;
  }
  public void setInputPath(Path inputPath) {
    this.inputPath = inputPath;
  }
  public int getRows() {
    return rows;
  }
  public void setRows(int rows) {
    this.rows = rows;
  }
  public int getCols() {
    return cols;
  }
  public void setCols(int cols) {
    this.cols = cols;
  }

}
