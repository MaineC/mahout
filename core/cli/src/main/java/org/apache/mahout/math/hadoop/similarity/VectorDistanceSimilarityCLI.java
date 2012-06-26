package org.apache.mahout.math.hadoop.similarity;

import static org.apache.mahout.math.hadoop.similarity.VectorDistanceConfig.OUT_TYPE_KEY;
import static org.apache.mahout.math.hadoop.similarity.VectorDistanceConfig.SEEDS;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractCLI;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;

public class VectorDistanceSimilarityCLI extends AbstractCLI {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new VectorDistanceSimilarityCLI(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(SEEDS, "s", "The set of vectors to compute distances against.  Must fit in memory on the mapper");
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(OUT_TYPE_KEY, "ot",
              "[pw|v] -- Define the output style: pairwise, the default, (pw) or vector (v).  Pairwise is a "
                  + "tuple of <seed, other, distance>, vector is <other, <Vector of size the number of seeds>>.",
              "pw");
    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    Path seeds = new Path(getOption(SEEDS));
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    if (measureClass == null) {
      measureClass = SquaredEuclideanDistanceMeasure.class.getName();
    }
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);
    if (getConf() == null) {
      setConf(new Configuration());
    }
    String outType = getOption(OUT_TYPE_KEY);
    if (outType == null) {
      outType = "pw";
    }

    VectorDistanceSimilarityJob job = new VectorDistanceSimilarityJob();
    job.run(getConf(), input, seeds, output, measure, outType);
    return 0;
  }

}
