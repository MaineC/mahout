package org.apache.mahout.math.hadoop.decomposer;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;

public class DistributedLanczosSolverCLI extends AbstractCLI {

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption("numRows", "nr", "Number of rows of the input matrix");
    addOption("numCols", "nc", "Number of columns of the input matrix");
    addOption("rank", "r", "Desired decomposition rank (note: only roughly 1/4 to 1/3 "
        + "of these will have the top portion of the spectrum)");
    addOption("symmetric", "sym", "Is the input matrix square and symmetric?");
    addOption("workingDir", "wd", "Working directory path to store Lanczos basis vectors "
                                  + "(to be used on restarts, and to avoid too much RAM usage)");
    // options required to run cleansvd job
    addOption("cleansvd", "cl", "Run the EigenVerificationJob to clean the eigenvectors after SVD", false);
    addOption("maxError", "err", "Maximum acceptable error", "0.05");
    addOption("minEigenvalue", "mev", "Minimum eigenvalue to keep the vector for", "0.0");
    addOption("inMemory", "mem", "Buffer eigen matrix into memory (if you have enough!)", "false");

    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    } else {
      Path inputPath = new Path(AbstractCLI.getOption(parsedArgs, "--input"));
      Path outputPath = new Path(AbstractCLI.getOption(parsedArgs, "--output"));
      Path outputTmpPath = new Path(AbstractCLI.getOption(parsedArgs, "--tempDir"));
      Path workingDirPath = AbstractCLI.getOption(parsedArgs, "--workingDir") != null
                          ? new Path(AbstractCLI.getOption(parsedArgs, "--workingDir")) : null;
      int numRows = Integer.parseInt(AbstractCLI.getOption(parsedArgs, "--numRows"));
      int numCols = Integer.parseInt(AbstractCLI.getOption(parsedArgs, "--numCols"));
      boolean isSymmetric = Boolean.parseBoolean(AbstractCLI.getOption(parsedArgs, "--symmetric"));
      int desiredRank = Integer.parseInt(AbstractCLI.getOption(parsedArgs, "--rank"));
  
      boolean cleansvd = Boolean.parseBoolean(AbstractCLI.getOption(parsedArgs, "--cleansvd"));
      if (cleansvd) {
        double maxError = Double.parseDouble(AbstractCLI.getOption(parsedArgs, "--maxError"));
        double minEigenvalue = Double.parseDouble(AbstractCLI.getOption(parsedArgs, "--minEigenvalue"));
        boolean inMemory = Boolean.parseBoolean(AbstractCLI.getOption(parsedArgs, "--inMemory"));
        return (new DistributedLanczosSolver()).runClean(inputPath,
                   outputPath,
                   outputTmpPath,
                   workingDirPath,
                   numRows,
                   numCols,
                   isSymmetric,
                   desiredRank,
                   maxError,
                   minEigenvalue,
                   inMemory);
      }
      DistributedLanczosSolver solver = new DistributedLanczosSolver();
      return solver.run(inputPath, outputPath, outputTmpPath, workingDirPath, numRows, numCols, isSymmetric, desiredRank);
    }
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new DistributedLanczosSolverCLI(), args);
  }

}
