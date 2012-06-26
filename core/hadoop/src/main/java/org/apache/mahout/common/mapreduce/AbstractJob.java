package org.apache.mahout.common.mapreduce;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;

public abstract class AbstractJob extends Configured {
  private static final Logger log = LoggerFactory.getLogger(AbstractJob.class);

  /** temp path, populated by {@link #parseArguments(String[]) */
  protected Path tempPath;

  protected Job prepareJob(Path inputPath,
      Path outputPath,
      Class<? extends InputFormat> inputFormat,
      Class<? extends Mapper> mapper,
      Class<? extends Writable> mapperKey,
      Class<? extends Writable> mapperValue,
      Class<? extends OutputFormat> outputFormat) throws IOException {

  Job job = HadoopUtil.prepareJob(inputPath, outputPath,
  inputFormat, mapper, mapperKey, mapperValue, outputFormat, getConf());
  job.setJobName(HadoopUtil.getCustomJobName(getClass().getSimpleName(), job, mapper, Reducer.class));
  return job;
  
  }

  protected Job prepareJob(Path inputPath, Path outputPath, Class<? extends Mapper> mapper,
      Class<? extends Writable> mapperKey, Class<? extends Writable> mapperValue, Class<? extends Reducer> reducer,
      Class<? extends Writable> reducerKey, Class<? extends Writable> reducerValue) throws IOException {
    return prepareJob(inputPath, outputPath, SequenceFileInputFormat.class, mapper, mapperKey, mapperValue, reducer,
        reducerKey, reducerValue, SequenceFileOutputFormat.class);
  }

  protected Job prepareJob(Path inputPath,
                           Path outputPath,
                           Class<? extends InputFormat> inputFormat,
                           Class<? extends Mapper> mapper,
                           Class<? extends Writable> mapperKey,
                           Class<? extends Writable> mapperValue,
                           Class<? extends Reducer> reducer,
                           Class<? extends Writable> reducerKey,
                           Class<? extends Writable> reducerValue,
                           Class<? extends OutputFormat> outputFormat) throws IOException {
    Job job = HadoopUtil.prepareJob(inputPath, outputPath,
            inputFormat, mapper, mapperKey, mapperValue, reducer, reducerKey, reducerValue, outputFormat, getConf());
    job.setJobName(HadoopUtil.getCustomJobName(getClass().getSimpleName(), job, mapper, Reducer.class));
    return job;
  }

  protected static boolean shouldRunNextPhase(Integer startPhase, Integer endPhase, AtomicInteger currentPhase) {
    int phase = currentPhase.getAndIncrement();
    boolean phaseSkipped = (startPhase != null && phase < startPhase)
        || (endPhase != null && phase > endPhase);
    if (phaseSkipped) {
      log.info("Skipping phase {}", phase);
    }
    return !phaseSkipped;
  }

  /**
   * Get the cardinality of the input vectors
   *
   * @param matrix
   * @return the cardinality of the vector
   */
  public int getDimensions(Path matrix) throws IOException, InstantiationException, IllegalAccessException {

    SequenceFile.Reader reader = null;
    try {
      reader = new SequenceFile.Reader(FileSystem.get(getConf()), matrix, getConf());

      Writable row = (Writable) reader.getKeyClass().newInstance();
      VectorWritable vectorWritable = new VectorWritable();

      Preconditions.checkArgument(reader.getValueClass().equals(VectorWritable.class),
          "value type of sequencefile must be a VectorWritable");

      boolean hasAtLeastOneRow = reader.next(row, vectorWritable);
      Preconditions.checkState(hasAtLeastOneRow, "matrix must have at least one row");

      return vectorWritable.get().size();

    } finally {
      Closeables.closeQuietly(reader);
    }
  }

  /**
   * necessary to make this job (having a combined input path) work on Amazon S3, hopefully this is obsolete when MultipleInputs is available
   * again
   */
  public static void setS3SafeCombinedInputPath(Job job, Path referencePath, Path inputPathOne, Path inputPathTwo)
      throws IOException {
    FileSystem fs = FileSystem.get(referencePath.toUri(), job.getConfiguration());
    FileInputFormat.setInputPaths(job, inputPathOne.makeQualified(fs), inputPathTwo.makeQualified(fs));
  }


  protected Path getTempPath() {
    return tempPath;
  }

  protected Path getTempPath(String directory) {
    return new Path(tempPath, directory);
  }
  
  public void setTempPath(Path temp) {
    this.tempPath = temp;
  }
}
