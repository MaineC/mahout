package org.apache.mahout.test;

import java.util.List;

import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;

import com.google.common.collect.Lists;

public class TasteTestUtils {

  public static DataModel getDataModel(long[] userIDs, Double[][] prefValues) {
    FastByIDMap<PreferenceArray> result = new FastByIDMap<PreferenceArray>();
    for (int i = 0; i < userIDs.length; i++) {
      List<Preference> prefsList = Lists.newArrayList();
      for (int j = 0; j < prefValues[i].length; j++) {
        if (prefValues[i][j] != null) {
          prefsList.add(new GenericPreference(userIDs[i], j, prefValues[i][j].floatValue()));
        }
      }
      if (!prefsList.isEmpty()) {
        result.put(userIDs[i], new GenericUserPreferenceArray(prefsList));
      }
    }
    return new GenericDataModel(result);
  }

  public static DataModel getDataModel() {
    return getDataModel(
            new long[] {1, 2, 3, 4},
            new Double[][] {
                    {0.1, 0.3},
                    {0.2, 0.3, 0.3},
                    {0.4, 0.3, 0.5},
                    {0.7, 0.3, 0.8},
            });
  }

  public static boolean arrayContains(long[] array, long value) {
    for (long l : array) {
      if (l == value) {
        return true;
      }
    }
    return false;
  }

}
