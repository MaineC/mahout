package org.apache.mahout.hadoop;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.math.hadoop.MatrixMultiplicationJob;
import org.apache.mahout.math.hadoop.MatrixConfig;

public class MatrixMultiplicationCLI extends AbstractCLI {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new MatrixMultiplicationCLI(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    addOption("numRowsA", "nra", "Number of rows of the first input matrix",
        true);
    addOption("numColsA", "nca", "Number of columns of the first input matrix",
        true);
    addOption("numRowsB", "nrb", "Number of rows of the second input matrix",
        true);

    addOption("numColsB", "ncb",
        "Number of columns of the second input matrix", true);
    addOption("inputPathA", "ia", "Path to the first input matrix", true);
    addOption("inputPathB", "ib", "Path to the second input matrix", true);

    Map<String, List<String>> argMap = parseArguments(strings);
    if (argMap == null) {
      return -1;
    }
    MatrixConfig configA = new MatrixConfig();
    configA.setInputPath(new Path(getOption("inputPathA")));
    configA.setRows(Integer.parseInt(getOption("numRowsA")));
    configA.setCols(Integer.parseInt(getOption("numColsA")));
    
    Path temp = new Path(getOption("tempDir"));
    
    MatrixConfig configB = new MatrixConfig();
    configB.setInputPath(new Path(getOption("inputPathB")));
    configB.setRows(Integer.parseInt(getOption("numRowsB")));
    configB.setCols(Integer.parseInt(getOption("numColsB")));
    MatrixMultiplicationJob job = new MatrixMultiplicationJob();
    return job.run(configA, configB, temp);
  }
}
