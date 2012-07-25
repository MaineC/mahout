package org.apache.mahout.math.hadoop.solver;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.math.solver.ConjugateGradientSolver;

public class DistributedConjugateGradientSolverCLI extends AbstractCLI implements Tool {
  
  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption("numRows", "nr", "Number of rows in the input matrix", true);
    addOption("numCols", "nc", "Number of columns in the input matrix", true);
    addOption("vector", "b", "Vector to solve against", true);
    addOption("lambda", "l", "Scalar in A + lambda * I [default = 0]", "0.0");
    addOption("symmetric", "sym", "Is the input matrix square and symmetric?", "true");
    addOption("maxIter", "x", "Maximum number of iterations to run");
    addOption("maxError", "err", "Maximum residual error to allow before stopping");

    Map<String, List<String>> parsedArgs = parseArguments(args);

    if (parsedArgs == null) {
      return -1;
    } else {
      Path inputPath = new Path(AbstractCLI.getOption(parsedArgs, "--input"));
      Path outputPath = new Path(AbstractCLI.getOption(parsedArgs, "--output"));
      Path tempPath = new Path(AbstractCLI.getOption(parsedArgs, "--tempDir"));
      Path vectorPath = new Path(AbstractCLI.getOption(parsedArgs, "--vector"));
      int numRows = Integer.parseInt(AbstractCLI.getOption(parsedArgs, "--numRows"));
      int numCols = Integer.parseInt(AbstractCLI.getOption(parsedArgs, "--numCols"));
      int maxIterations = parsedArgs.containsKey("--maxIter") ? Integer.parseInt(AbstractCLI.getOption(parsedArgs, "--maxIter")) : numCols;
      double maxError = parsedArgs.containsKey("--maxError") 
          ? Double.parseDouble(AbstractCLI.getOption(parsedArgs, "--maxError"))
          : ConjugateGradientSolver.DEFAULT_MAX_ERROR;


      setConf(new Configuration());
      return (new DistributedConjugateGradientSolver().run(inputPath, vectorPath, tempPath,
          numRows, numCols, maxIterations, maxError, outputPath));
    }
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new DistributedConjugateGradientSolverCLI(), args);
  }
}
