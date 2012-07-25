package org.apache.mahout.common;

public class MahoutHadoopTestCase extends MahoutTestCase {
  /**
   * @return a job option key string (--name) from the given option name
   */
  protected static String optKey(String optionName) {
    return AbstractCLI.keyFor(optionName);
  }
}
