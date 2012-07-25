package org.apache.mahout.hadoop;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.math.hadoop.MatrixConfig;
import org.apache.mahout.math.hadoop.TransposeJob;

public class TransposeCLI extends AbstractCLI {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new TransposeCLI(), args);
  }


  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOption("numRows", "nr", "Number of rows of the input matrix");
    addOption("numCols", "nc", "Number of columns of the input matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }
  
    int numRows = Integer.parseInt(getOption("numRows"));
    int numCols = Integer.parseInt(getOption("numCols"));

    MatrixConfig config = new MatrixConfig();
    config.setInputPath(getInputPath());
    config.setRows(numRows);
    config.setCols(numCols);
    TransposeJob job = new TransposeJob();
    return job.run(config, getTempPath());
  }
}
